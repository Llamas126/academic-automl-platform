"use client";

import { useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle2, Download, Loader2, XCircle } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button, buttonVariants } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { downloadResultsZip, saveBlob } from "@/lib/api/endpoints";
import type { JobState } from "@/lib/api/types";
import { useJobStatus } from "@/hooks/use-job-status";
import { useI18n } from "@/lib/i18n";

const FLOW: JobState[] = ["PENDING", "STARTED", "PROGRESS", "SUCCESS"];

function StateBadge({ state }: { state: JobState }) {
  const { t } = useI18n();
  const variant =
    state === "SUCCESS" ? "default" : state === "FAILURE" ? "destructive" : "secondary";
  return <Badge variant={variant}>{t(`jobs.state.${state}`)}</Badge>;
}

export default function JobPage() {
  const { t } = useI18n();
  const params = useParams<{ id: string }>();
  const jobId = params.id;
  const queryClient = useQueryClient();
  const { data: job, isError, refetch } = useJobStatus(jobId);
  const [downloading, setDownloading] = useState(false);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!job || job.state === "SUCCESS" || job.state === "FAILURE") return;
    const timer = setInterval(() => setElapsed((value) => value + 1), 1_000);
    return () => clearInterval(timer);
  }, [job]);

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ["results", jobId] });

  const handleDownload = async () => {
    setDownloading(true);
    try {
      const { blob, filename } = await downloadResultsZip(jobId);
      saveBlob(blob, filename);
    } catch (error) {
      toast.error(t("common.error"), { description: String(error) });
    } finally {
      setDownloading(false);
    }
  };

  if (isError) {
    return (
      <Card className="mx-auto max-w-md">
        <CardContent className="flex flex-col items-center gap-3 py-8 text-center">
          <XCircle className="h-8 w-8 text-destructive" />
          <p>{t("jobs.notFound")}</p>
          <Button variant="outline" onClick={() => void refetch()}>{t("common.retry")}</Button>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">{t("jobs.title")}</h1>
        {job && <StateBadge state={job.state} />}
      </div>

      {!job ? (
        <div className="space-y-3">
          <Skeleton className="h-4 w-40" />
          <Skeleton className="h-24 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
      ) : (
        <>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="font-mono text-sm font-normal text-muted-foreground">{job.job_id}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">{t("jobs.progress")}</span>
                  <span className="font-semibold tabular-nums">{Math.round(job.progress * 100)}%</span>
                </div>
                <Progress
                  value={job.state === "FAILURE" ? 100 : Math.max(2, job.progress * 100)}
                  aria-label={t("jobs.progress")}
                />
              </div>

              {job.stage && (
                <p className="text-sm">
                  <span className="text-muted-foreground">{t("jobs.stage")}: </span>
                  <span className="font-medium">{job.stage}</span>
                </p>
              )}

              {!["SUCCESS", "FAILURE"].includes(job.state) && (
                <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Loader2 className="h-3 w-3 animate-spin" /> {t("jobs.elapsed")}: {elapsed}s
                </p>
              )}

              {job.state === "FAILURE" && job.error && (
                <div className="flex items-start gap-2 rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
                  <div>
                    <p className="font-medium text-destructive">{t("jobs.failedDetail")}</p>
                    <p className="mt-0.5 break-all font-mono text-xs">{job.error}</p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <ol className="flex items-center justify-between px-2 text-[11px] text-muted-foreground">
            {FLOW.map((state, index) => {
              const currentIndex = job.state === "FAILURE" ? 2 : FLOW.indexOf(job.state);
              const reached = index <= currentIndex;
              return (
                <li key={state} className="flex flex-1 flex-col items-center gap-1">
                  <span
                    className={`flex h-6 w-6 items-center justify-center rounded-full ${
                      reached ? "bg-primary text-primary-foreground" : "bg-muted"
                    }`}
                  >
                    {state === "SUCCESS" && job.state === "SUCCESS" ? (
                      <CheckCircle2 className="h-3.5 w-3.5" />
                    ) : (
                      index + 1
                    )}
                  </span>
                  {t(`jobs.state.${state}`)}
                </li>
              );
            })}
          </ol>

          {job.state === "SUCCESS" && (
            <div className="flex flex-wrap justify-end gap-2">
              <Button variant="outline" onClick={handleDownload} disabled={downloading}>
                <Download className="mr-1 h-4 w-4" />
                {downloading ? t("results.downloading") : t("jobs.downloadZip")}
              </Button>
              <Link href={`/results/${jobId}`} onClick={invalidate} className={buttonVariants()}>
                {t("jobs.viewResults")}
              </Link>
            </div>
          )}

          {["FAILURE", "SUCCESS"].includes(job.state) && (
            <div className="flex justify-start">
              <Link href="/" className={buttonVariants({ variant: "ghost", size: "sm" })}>
                {t("jobs.newTrain")}
              </Link>
            </div>
          )}
        </>
      )}
    </div>
  );
}
