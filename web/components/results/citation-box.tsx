"use client";

import { Check, Copy } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import type { RunSummary } from "@/lib/api/types";
import { useI18n } from "@/lib/i18n";

function buildBibtex(summary: RunSummary, target: string, model: string): string {
  const year = new Date().getFullYear();
  const citeKey = `automl_${summary.job_id.slice(0, 8)}_${year}`;
  return [
    `@misc{${citeKey},`,
    `  title       = {Academic AutoML: ${model} model for target '${target}'},`,
    `  author      = {{Academic AutoML Platform}},`,
    `  year        = {${year}},`,
    `  note        = {Automated ML run ${summary.job_id}; dataset '${summary.filename}'; ${summary.n_samples} samples; ${Object.entries(summary.library_versions).map(([lib, version]) => `${lib}=${version}`).join(", ")}}`,
    "}",
  ].join("\n");
}

export function CitationBox({ summary, target }: { summary: RunSummary; target: string }) {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);
  const bestModel =
    summary.targets.find((entry) => entry.target === target)?.best_model ?? summary.targets[0]?.best_model ?? "";
  const bibtex = buildBibtex(summary, target, bestModel);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(bibtex);
    setCopied(true);
    toast.success(t("results.copied"));
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm">{t("results.cite")}</CardTitle>
        <Button variant="outline" size="sm" onClick={handleCopy}>
          {copied ? <Check className="mr-1 h-3.5 w-3.5" /> : <Copy className="mr-1 h-3.5 w-3.5" />}
          {t("results.copyBibtex")}
        </Button>
      </CardHeader>
      <CardContent>
        <pre className="overflow-auto rounded-md bg-muted p-3 font-mono text-[11px] leading-relaxed">{bibtex}</pre>
      </CardContent>
    </Card>
  );
}
