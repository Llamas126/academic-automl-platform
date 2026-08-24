"use client";

import { Download } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { PredictResponse } from "@/lib/api/types";
import { saveBlob } from "@/lib/api/endpoints";
import { useI18n } from "@/lib/i18n";

function flattenRow(row: Record<string, unknown>): Record<string, string> {
  const flat: Record<string, string> = {};
  for (const [key, value] of Object.entries(row)) {
    if (value !== null && typeof value === "object") {
      for (const [innerKey, innerValue] of Object.entries(value as Record<string, unknown>)) {
        flat[`${key}.${innerKey}`] = String(innerValue);
      }
    } else {
      flat[key] = String(value);
    }
  }
  return flat;
}

export function PredictionsGrid({ result }: { result: PredictResponse }) {
  const { t } = useI18n();
  const rows = result.predictions.map(flattenRow);
  const columns = Array.from(new Set(rows.flatMap((row) => Object.keys(row))));

  const exportCsv = () => {
    const escape = (value: string) => (/[",\n]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value);
    const csv = [
      columns.join(","),
      ...rows.map((row) => columns.map((column) => escape(row[column] ?? "")).join(",")),
    ].join("\n");
    saveBlob(new Blob([csv], { type: "text/csv;charset=utf-8" }), `predictions_${result.job_id}.csv`);
  };

  return (
    <Card>
      <CardHeader className="flex-row flex-wrap items-center justify-between gap-2 space-y-0 pb-2">
        <CardTitle className="text-sm">{t("predict.resultsFor", { n: result.n_rows })}</CardTitle>
        <Button variant="outline" size="sm" onClick={exportCsv}>
          <Download className="mr-1 h-3.5 w-3.5" /> {t("predict.downloadCsv")}
        </Button>
      </CardHeader>
      <CardContent>
        <div className="max-h-[420px] overflow-auto rounded-md border">
          <table className="w-full text-xs">
            <thead className="sticky top-0 bg-muted">
              <tr>
                {columns.map((column) => (
                  <th key={column} className="whitespace-nowrap px-2 py-1.5 text-left font-mono font-medium">
                    {column.includes(".") ? (
                      <span>
                        {column.split(".")[0]}
                        <span className="ml-1 rounded bg-primary/10 px-1 text-[9px] uppercase">
                          {t("predict.probabilities")}
                        </span>
                      </span>
                    ) : (
                      column === "" || column
                    )}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr key={rowIndex} className={rowIndex % 2 ? "bg-muted/30" : ""}>
                  {columns.map((column) => (
                    <td key={column} className="whitespace-nowrap px-2 py-1.5 tabular-nums">
                      {row[column] ?? ""}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {Object.keys(result.targets).length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2 text-xs text-muted-foreground">
            {Object.values(result.targets).map((meta) => (
              <span key={meta.target} className="rounded border px-1.5 py-0.5">
                {meta.target}: {meta.model} ({meta.task})
              </span>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
