"use client";

import { useMutation } from "@tanstack/react-query";
import { FileSpreadsheet, Plus, Search, X } from "lucide-react";
import { useSearchParams } from "next/navigation";
import { Suspense, useRef, useState } from "react";
import { toast } from "sonner";

import { PredictionsGrid } from "@/components/predict/predictions-grid";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ApiError } from "@/lib/api/client";
import { predictFile, predictRecords } from "@/lib/api/endpoints";
import type { PredictResponse } from "@/lib/api/types";
import { useI18n } from "@/lib/i18n";
import { useResults } from "@/hooks/use-results";

function PredictPageInner() {
  const { t } = useI18n();
  const searchParams = useSearchParams();
  const [jobInput, setJobInput] = useState(searchParams.get("job") ?? "");
  const [jobId, setJobId] = useState<string | null>(null);
  const [mode, setMode] = useState<"records" | "file">("records");
  const [rows, setRows] = useState<Record<string, string>[]>([]);
  const [selectedTargets, setSelectedTargets] = useState<string[]>([]);
  const [file, setFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);

  const modelQuery = useResults(jobId ?? "");

  const recordsMutation = useMutation({
    mutationFn: (parsedRows: Record<string, unknown>[]) => predictRecords(jobId!, parsedRows),
    onSuccess: (response) => setPrediction(response),
    onError: (error: ApiError) => toast.error(t("common.error"), { description: error.detail }),
  });

  const fileMutation = useMutation({
    mutationFn: (selected: File) =>
      predictFile(jobId!, selected, selectedTargets.length ? selectedTargets : undefined),
    onSuccess: (response) => setPrediction(response),
    onError: (error: ApiError) => toast.error(t("common.error"), { description: error.detail }),
  });

  const loadModel = () => {
    const candidate = jobInput.trim();
    if (!candidate) return;
    setPrediction(null);
    setJobId(candidate);
  };

  const summary = modelQuery.data;
  const featureColumns = summary?.feature_columns ?? [];

  const addRow = () => {
    const empty: Record<string, string> = {};
    for (const column of featureColumns) empty[column] = "";
    setRows((current) => [...current, empty]);
  };

  const updateCell = (rowIndex: number, column: string, value: string) => {
    setRows((current) =>
      current.map((row, index) => (index === rowIndex ? { ...row, [column]: value } : row)),
    );
  };

  const submitRecords = () => {
    if (rows.length === 0) return;
    const parsed = rows.map((row) => {
      const out: Record<string, unknown> = {};
      for (const [column, raw] of Object.entries(row)) {
        const trimmed = raw.trim();
        if (trimmed === "") continue;
        const asNumber = Number(trimmed);
        out[column] =
          Number.isFinite(asNumber) && /^-?[\d.]+(e-?\d+)?$/i.test(trimmed) ? asNumber : trimmed;
      }
      return out;
    });
    recordsMutation.mutate(parsed);
  };

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      <h1 className="text-xl font-semibold">{t("predict.title")}</h1>

      <Card>
        <CardContent className="flex flex-col gap-2 py-4 sm:flex-row sm:items-end">
          <div className="flex-1 space-y-1">
            <p className="text-xs text-muted-foreground">{t("predict.jobId")}</p>
            <Input
              value={jobInput}
              onChange={(event) => setJobInput(event.target.value)}
              placeholder="a1b2c3d4-…"
              onKeyDown={(event) => event.key === "Enter" && loadModel()}
              className="font-mono"
            />
          </div>
          <Button onClick={loadModel} disabled={modelQuery.isLoading}>
            <Search className="mr-1 h-4 w-4" />
            {modelQuery.isLoading ? t("predict.loadingModel") : t("predict.load")}
          </Button>
        </CardContent>
      </Card>

      {jobId && modelQuery.isLoading && (
        <div className="space-y-2">
          <Skeleton className="h-4 w-48" />
          <Skeleton className="h-40 w-full rounded-xl" />
        </div>
      )}

      {jobId && modelQuery.isError && (
        <Card>
          <CardContent className="py-4 text-sm text-destructive">
            {(modelQuery.error as ApiError)?.detail ?? t("common.error")}
          </CardContent>
        </Card>
      )}

      {summary && (
        <>
          <Tabs value={mode} onValueChange={(value) => setMode(value as "records" | "file")}>
            <TabsList>
              <TabsTrigger value="records">{t("predict.mode.records")}</TabsTrigger>
              <TabsTrigger value="file">{t("predict.mode.file")}</TabsTrigger>
            </TabsList>
          </Tabs>

          {mode === "records" && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">{t("predict.mode.records")}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {rows.length === 0 ? (
                  <Button variant="outline" onClick={addRow}>
                    <Plus className="mr-1 h-4 w-4" /> {t("predict.addRow")}
                  </Button>
                ) : (
                  <>
                    <div className="overflow-auto rounded-md border">
                      <table className="w-full text-xs">
                        <thead className="bg-muted">
                          <tr>
                            {featureColumns.map((column) => (
                              <th key={column} className="whitespace-nowrap px-2 py-1.5 text-left font-mono font-medium">
                                {column}
                              </th>
                            ))}
                            <th className="w-8" />
                          </tr>
                        </thead>
                        <tbody>
                          {rows.map((row, rowIndex) => (
                            <tr key={rowIndex} className="border-t">
                              {featureColumns.map((column) => (
                                <td key={column} className="px-1 py-1">
                                  <Input
                                    value={row[column] ?? ""}
                                    onChange={(event) => updateCell(rowIndex, column, event.target.value)}
                                    className="h-7 border-transparent bg-transparent px-1.5 font-mono text-xs focus-visible:bg-background"
                                  />
                                </td>
                              ))}
                              <td className="px-1">
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-7 w-7"
                                  aria-label={t("predict.removeRow")}
                                  onClick={() => setRows((current) => current.filter((_, index) => index !== rowIndex))}
                                >
                                  <X className="h-3.5 w-3.5" />
                                </Button>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    <div className="flex flex-wrap justify-between gap-2">
                      <Button variant="outline" size="sm" onClick={addRow}>
                        <Plus className="mr-1 h-3.5 w-3.5" /> {t("predict.addRow")}
                      </Button>
                      <Button
                        size="sm"
                        disabled={recordsMutation.isPending}
                        onClick={submitRecords}
                      >
                        {recordsMutation.isPending ? t("common.loading") : t("predict.runRecords")}
                      </Button>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          )}

          {mode === "file" && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm">{t("predict.mode.file")}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {summary.targets.length > 0 && (
                  <div className="space-y-1.5">
                    <p className="text-xs text-muted-foreground">{t("config.targets")}</p>
                    <div className="flex flex-wrap gap-2">
                      {summary.targets.map((target) => {
                        const active = selectedTargets.includes(target.target);
                        return (
                          <Badge
                            key={target.target}
                            variant={active ? "default" : "outline"}
                            className="cursor-pointer hover:bg-accent"
                            onClick={() =>
                              setSelectedTargets((current) =>
                                active
                                  ? current.filter((name) => name !== target.target)
                                  : [...current, target.target],
                              )
                            }
                          >
                            {target.target}
                          </Badge>
                        );
                      })}
                    </div>
                  </div>
                )}

                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={(event) => {
                    event.preventDefault();
                    const dropped = event.dataTransfer.files[0];
                    if (dropped) setFile(dropped);
                  }}
                  className="flex min-h-32 w-full flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-6 text-center text-sm transition-colors hover:border-primary/50"
                >
                  <FileSpreadsheet className="h-8 w-8 text-muted-foreground" />
                  {file ? file.name : t("predict.dropFile")}
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.xlsx"
                  className="hidden"
                  onChange={(event) => {
                    const picked = event.target.files?.[0];
                    if (picked) setFile(picked);
                    event.target.value = "";
                  }}
                />

                <div className="flex justify-end">
                  <Button
                    disabled={!file || fileMutation.isPending}
                    onClick={() => file && fileMutation.mutate(file)}
                  >
                    {fileMutation.isPending ? t("common.loading") : t("predict.runFile")}
                  </Button>
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}

      {prediction && <PredictionsGrid result={prediction} />}
    </div>
  );
}

export default function PredictPage() {
  return (
    <Suspense fallback={<Skeleton className="mx-auto mt-10 h-48 w-full max-w-4xl rounded-xl" />}>
      <PredictPageInner />
    </Suspense>
  );
}
