"use client";

import { AlertTriangle, Download, FlaskConical } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";
import { toast } from "sonner";

import { CitationBox } from "@/components/results/citation-box";
import { ComparisonChart, ModelsTable } from "@/components/results/models-table";
import { ShapGallery } from "@/components/results/shap-gallery";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useFigures, useResults } from "@/hooks/use-results";
import { downloadResultsZip, saveBlob } from "@/lib/api/endpoints";
import { ApiError } from "@/lib/api/client";
import { useI18n } from "@/lib/i18n";

export default function ResultsPage() {
  const { t } = useI18n();
  const params = useParams<{ id: string }>();
  const jobId = params.id;
  const { data: summary, isLoading, isError, error, refetch } = useResults(jobId);
  const { data: figures, isLoading: figuresLoading } = useFigures(jobId);
  const [downloading, setDownloading] = useState(false);
  const [activeTarget, setActiveTarget] = useState<string | null>(null);

  if (isError) {
    return (
      <Card className="mx-auto max-w-md">
        <CardContent className="flex flex-col items-center gap-3 py-8 text-center">
          <AlertTriangle className="h-8 w-8 text-destructive" />
          <p className="text-sm text-muted-foreground">
            {(error as ApiError)?.detail ?? t("common.error")}
          </p>
          <Button variant="outline" onClick={() => void refetch()}>{t("common.retry")}</Button>
        </CardContent>
      </Card>
    );
  }

  if (isLoading || !summary) {
    return (
      <div className="space-y-4">
        <div className="grid gap-3 sm:grid-cols-4">
          {[0, 1, 2, 3].map((index) => (
            <Skeleton key={index} className="h-20 rounded-xl" />
          ))}
        </div>
        <Skeleton className="h-72 w-full rounded-xl" />
      </div>
    );
  }

  const currentTarget =
    activeTarget && summary.targets.some((entry) => entry.target === activeTarget)
      ? activeTarget
      : summary.targets[0]?.target ?? "";
  const targetResult = summary.targets.find((entry) => entry.target === currentTarget);
  const targetFigures = (figures?.figures ?? []).filter(
    (figure) => figure === `${currentTarget}` || figure.startsWith(`${currentTarget}/`),
  );

  const handleDownload = async () => {
    setDownloading(true);
    try {
      const { blob, filename } = await downloadResultsZip(jobId);
      saveBlob(blob, filename);
    } catch (caught) {
      toast.error(t("common.error"), { description: (caught as ApiError).detail });
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-xl font-semibold">{t("results.title")}</h1>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" size="sm" render={<Link href={`/predict?job=${jobId}`} />}>
            <FlaskConical className="mr-1 h-4 w-4" /> {t("results.predictCta")}
          </Button>
          <Button size="sm" onClick={handleDownload} disabled={downloading}>
            <Download className="mr-1 h-4 w-4" />
            {downloading ? t("results.downloading") : t("results.downloadZip")}
          </Button>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {[
          { label: t("results.dataset"), value: summary.filename },
          { label: t("results.samples"), value: summary.n_samples.toLocaleString() },
          { label: t("results.features"), value: String(summary.n_features) },
          { label: t("results.runtime"), value: `${summary.runtime_s.toFixed(1)}s` },
        ].map((item) => (
          <Card key={item.label}>
            <CardHeader className="pb-1">
              <CardTitle className="text-xs font-medium text-muted-foreground">{item.label}</CardTitle>
            </CardHeader>
            <CardContent className="truncate text-sm font-semibold">{item.value}</CardContent>
          </Card>
        ))}
      </div>

      {summary.warnings.length > 0 && (
        <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 p-3 text-sm">
          <p className="mb-1 flex items-center gap-1.5 font-medium text-amber-600 dark:text-amber-400">
            <AlertTriangle className="h-4 w-4" /> {t("results.warnings")}
          </p>
          <ul className="list-inside list-disc space-y-0.5 text-xs text-muted-foreground">
            {summary.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      <Tabs value={currentTarget} onValueChange={setActiveTarget}>
        <TabsList className="flex-wrap">
          {summary.targets.map((entry) => (
            <TabsTrigger key={entry.target} value={entry.target} className="gap-1.5">
              {entry.target}
              <Badge variant="outline" className="text-[10px]">{entry.task}</Badge>
            </TabsTrigger>
          ))}
        </TabsList>

        {summary.targets.map((entry) => (
          <TabsContent key={entry.target} value={entry.target} className="space-y-4 pt-4">
            <Card>
              <CardHeader className="pb-2">
                <div className="flex flex-wrap items-center gap-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">{t("results.bestModel")}</CardTitle>
                  <Badge>{entry.best_model}</Badge>
                </div>
              </CardHeader>
              {Object.keys(entry.best_params).length > 0 && (
                <CardContent>
                  <p className="mb-1 text-xs text-muted-foreground">{t("results.bestParams")}</p>
                  <pre className="max-h-28 overflow-auto rounded-md bg-muted p-2 font-mono text-[11px]">
                    {JSON.stringify(entry.best_params, null, 2)}
                  </pre>
                </CardContent>
              )}
            </Card>

            <Tabs defaultValue="overview">
              <TabsList>
                <TabsTrigger value="overview">{t("results.tab.overview")}</TabsTrigger>
                <TabsTrigger value="shap">{t("results.tab.shap")}</TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="space-y-4 pt-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm">
                      {t("results.metric")} · {targetResult?.models[0]?.primary_metric ?? ""}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ComparisonChart models={entry.models} />
                  </CardContent>
                </Card>
                <Card>
                  <CardContent className="p-0">
                    <ModelsTable models={entry.models} />
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="shap" className="pt-4">
                <ShapGallery jobId={jobId} figures={targetFigures} loading={figuresLoading} />
              </TabsContent>
            </Tabs>
          </TabsContent>
        ))}
      </Tabs>

      <CitationBox summary={summary} target={currentTarget} />
    </div>
  );
}
