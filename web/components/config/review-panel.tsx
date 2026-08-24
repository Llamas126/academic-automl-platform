"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { DatasetProfile } from "@/lib/api/types";
import type { TrainFormValues } from "@/components/config/config-form";
import { MODEL_CATALOG } from "@/lib/constants";
import { useI18n } from "@/lib/i18n";

export function ReviewPanel({ profile, values }: { profile: DatasetProfile; values: TrainFormValues }) {
  const { t } = useI18n();
  const modelLabels = values.models.length
    ? MODEL_CATALOG.filter((model) => values.models.includes(model.key)).map((model) => model.label)
    : [t("config.modelsAll")];

  const rows = [
    { label: t("review.dataset"), value: `${profile.filename} (${profile.n_rows}×${profile.n_cols})` },
    {
      label: t("review.targetsList"),
      value: values.targets.map((target) => `${target.name} [${t(`config.task.${target.task}`)}]`).join(", "),
    },
    { label: t("config.folds"), value: `${values.cv.n_folds} · ${t(`config.cvMode.${values.cv.mode}`)}${values.cv.mode === "nested" ? ` (${values.cv.n_inner_folds})` : ""}` },
    { label: t("config.hpo.enabled"), value: values.hpo.enabled ? `${t("common.of")} ${values.hpo.n_trials} ${t("config.trials").toLowerCase()} · ${values.hpo.timeout_s ?? t("config.timeoutOff")}` : "—" },
    { label: t("config.shap"), value: values.explain.shap ? `${values.explain.background_samples} / top-${values.explain.max_display}` : "—" },
    { label: t("config.dpi"), value: String(values.exportConfig.figures_dpi) },
  ];

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">{t("review.title")}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {rows.map((row) => (
            <div key={row.label} className="flex flex-col gap-1 border-b pb-3 last:border-b-0 last:pb-0 sm:flex-row sm:items-center sm:justify-between">
              <span className="text-sm text-muted-foreground">{row.label}</span>
              <span className="text-sm font-medium">{row.value}</span>
            </div>
          ))}
          <div className="flex flex-wrap gap-1.5 pt-1">
            {modelLabels.map((label) => (
              <Badge key={label} variant="secondary">{label}</Badge>
            ))}
          </div>
        </CardContent>
      </Card>
      <p className="text-center text-sm text-muted-foreground">{t("review.readyToLaunch")} 🚀</p>
    </div>
  );
}
