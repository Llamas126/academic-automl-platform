from __future__ import annotations

from datetime import date

import httpx
import pytest
from ingestion.domain.mango.nasa_api import (
    NASA_POWER_URL,
    NasaFetchError,
    build_nasa_params,
    fetch_nasa_power_csv,
    load_climate_source,
)
from ingestion.domain.mango.spec import NASA_PARAMETERS, MangoDatasetSpec

from tests.fixtures.make_fixtures import make_nasa_power_csv

SPEC = MangoDatasetSpec(date_start=date(2018, 3, 1), date_end=date(2026, 8, 30))


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def ok_handler(text: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=text)

    return handler


def test_build_nasa_params_uses_spec() -> None:
    params = build_nasa_params(SPEC)

    assert params["parameters"] == ",".join(NASA_PARAMETERS)
    assert params["community"] == "AG"
    assert params["latitude"] == "10.8500"
    assert params["longitude"] == "-74.7800"
    assert params["start"] == "20180301"
    assert params["end"] == "20260830"
    assert params["format"] == "CSV"


def test_fetch_retries_transient_failures_then_succeeds() -> None:
    text = make_nasa_power_csv(n_days=3)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if len(calls) < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, text=text)

    with mock_client(handler) as client:
        result = fetch_nasa_power_csv(SPEC, client=client, retries=3, backoff=0)

    assert result == text
    assert len(calls) == 3
    assert calls[0].startswith(NASA_POWER_URL)


def test_fetch_does_not_retry_client_errors() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, text="bad request")

    with mock_client(handler) as client:
        with pytest.raises(NasaFetchError, match="400"):
            fetch_nasa_power_csv(SPEC, client=client, retries=3, backoff=0)

    assert len(calls) == 1


def test_fetch_retries_invalid_body_then_fails() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(200, text="not a nasa response")

    with mock_client(handler) as client:
        with pytest.raises(NasaFetchError, match="after 3 attempts"):
            fetch_nasa_power_csv(SPEC, client=client, retries=3, backoff=0)

    assert len(calls) == 3


def test_fetch_retries_transport_errors() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("network down", request=request)

    with mock_client(handler) as client:
        with pytest.raises(NasaFetchError, match="network down"):
            fetch_nasa_power_csv(SPEC, client=client, retries=2, backoff=0)

    assert len(calls) == 2


def test_load_climate_source_prefers_api_when_online() -> None:
    text = make_nasa_power_csv(n_days=4)

    with mock_client(ok_handler(text)) as client:
        source = load_climate_source(SPEC, csv_path=None, online=True, client=client)

    assert source.origin == "nasa_api"
    assert source.climate.days == 4
    assert source.warnings == ()


def test_load_climate_source_falls_back_to_csv_on_api_failure(tmp_path) -> None:
    path = tmp_path / "nasa.csv"
    path.write_text(make_nasa_power_csv(n_days=5), encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down", request=request)

    with mock_client(handler) as client:
        source = load_climate_source(SPEC, csv_path=path, online=True, client=client, retries=2, backoff=0)

    assert source.origin == "local_csv"
    assert source.climate.days == 5
    assert len(source.warnings) == 1
    assert "using local CSV" in source.warnings[0]


def test_load_climate_source_offline_never_calls_the_api(tmp_path) -> None:
    path = tmp_path / "nasa.csv"
    path.write_text(make_nasa_power_csv(n_days=5), encoding="utf-8")
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500)

    with mock_client(handler) as client:
        source = load_climate_source(SPEC, csv_path=path, online=False, client=client)

    assert source.origin == "local_csv"
    assert source.warnings == ()
    assert calls == []


def test_load_climate_source_without_any_source_raises() -> None:
    with pytest.raises(NasaFetchError, match="no NASA POWER CSV configured"):
        load_climate_source(SPEC, csv_path=None, online=False)


def test_load_climate_source_raises_when_online_fails_and_file_is_missing(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down", request=request)

    with mock_client(handler) as client:
        with pytest.raises(NasaFetchError, match="not found"):
            load_climate_source(
                SPEC,
                csv_path=tmp_path / "missing.csv",
                online=True,
                client=client,
                retries=1,
                backoff=0,
            )
