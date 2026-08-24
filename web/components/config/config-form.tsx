"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { X } from "lucide-react";
import { useMemo } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import type { DatasetProfile } from "@/lib/api/types";
import { MODEL_CATALOG, SELECTABLE_ROLES } from "@/lib/constants";
import { useI18n } from "@/lib/i18n";

export const trainSchema = z.object({
  targets: z
    .array(
      z.object({
        name: z.string().min(1),
        task: z.enum(["auto", "classification", "regression"]),
      }),
    )
    .min(1),
  models: z.array(z.string()),
  cv: z.object({
    n_folds: z.number().int().min(2).max(20),
    mode: z.enum(["nested", "single"]),
    n_inner_folds: z.number().int().min(2).max(10),
    random_state: z.number().int(),
  }),
  hpo: z.object({
    enabled: z.boolean(),
    n_trials: z.number().int().min(1).max(500),
    timeout_s: z.number().int().min(1).nullable(),
  }),
  explain: z.object({
    shap: z.boolean(),
    background_samples: z.number().int().min(10).max(1000),
    max_display: z.number().int().min(3).max(50),
  }),
  exportConfig: z.object({
    figures_dpi: z.number().int().min(72).max(600),
    include_eps: z.boolean(),
    include_shap_dependence: z.boolean(),
  }),
});

export type TrainFormValues = z.infer<typeof trainSchema>;

export const DEFAULT_FORM: TrainFormValues = {
  targets: [],
  models: [],
  cv: { n_folds: 5, mode: "nested", n_inner_folds: 3, random_state: 42 },
  hpo: { enabled: true, n_trials: 25, timeout_s: 3600 },
  explain: { shap: true, background_samples: 100, max_display: 20 },
  exportConfig: { figures_dpi: 300, include_eps: true, include_shap_dependence: true },
};

function ConfigSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">{children}</CardContent>
    </Card>
  );
}

export function ConfigForm({
  profile,
  form,
}: {
  profile: DatasetProfile;
  form: ReturnType<typeof useForm<TrainFormValues>>;
}) {
  const { t } = useI18n();

  const selectable = useMemo(
    () =>
      profile.columns.filter(
        (column) => SELECTABLE_ROLES.has(column.role) && !column.excluded_from_features,
      ),
    [profile],
  );

  const selected = form.watch("targets");
  const cvMode = form.watch("cv.mode");
  const hpoEnabled = form.watch("hpo.enabled");
  const shapEnabled = form.watch("explain.shap");
  const models = form.watch("models");

  return (
    <Form {...form}>
      <form className="space-y-4">
        <ConfigSection title={t("config.targets")}>
          <p className="text-xs text-muted-foreground">{t("config.targetsHint")}</p>
          <div className="flex flex-wrap gap-2">
            {selectable.map((column) => {
              const active = selected.some((target) => target.name === column.name);
              return (
                <Badge
                  key={column.name}
                  variant={active ? "default" : "outline"}
                  className="cursor-pointer hover:bg-accent"
                  onClick={() =>
                    form.setValue(
                      "targets",
                      active ? selected.filter((target) => target.name !== column.name) : [...selected, { name: column.name, task: "auto" as const }],
                      { shouldValidate: true },
                    )
                  }
                >
                  {active ? "✓ " : ""}
                  {column.name}
                </Badge>
              );
            })}
          </div>

          {selected.length > 0 && (
            <div className="space-y-2">
              {selected.map((target, index) => (
                <div key={target.name} className="flex items-center gap-2">
                  <span className="w-40 truncate font-mono text-sm">{target.name}</span>
                  <FormField
                    control={form.control}
                    name={`targets.${index}.task`}
                    render={({ field }) => (
                      <Select value={field.value} onValueChange={field.onChange}>
                        <SelectTrigger className="w-44">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="auto">{t("config.task.auto")}</SelectItem>
                          <SelectItem value="classification">{t("config.task.classification")}</SelectItem>
                          <SelectItem value="regression">{t("config.task.regression")}</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    aria-label={t("predict.removeRow")}
                    onClick={() =>
                      form.setValue(
                        "targets",
                        selected.filter((entry) => entry.name !== target.name),
                        { shouldValidate: true },
                      )
                    }
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              ))}
            </div>
          )}
          <FormMessage>{selected.length === 0 ? t("config.targetsHint") : ""}</FormMessage>
        </ConfigSection>

        <ConfigSection title={t("config.models")}>
          <p className="text-xs text-muted-foreground">
            {models.length === 0 ? t("config.modelsAll") : `${models.length} ${t("results.model").toLowerCase()}s`}
          </p>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {MODEL_CATALOG.map((model) => (
              <label key={model.key} className="flex cursor-pointer items-center gap-2 rounded-md border p-2 text-sm hover:bg-accent/50">
                <Checkbox
                  checked={models.includes(model.key)}
                  onCheckedChange={(checked) =>
                    form.setValue(
                      "models",
                      checked ? [...models, model.key] : models.filter((key) => key !== model.key),
                    )
                  }
                />
                <span className="truncate">{model.label}</span>
                <Badge variant="outline" className="ml-auto shrink-0 text-[10px]">
                  {model.task === "regression" ? t("config.task.regression") : t("config.task.classification")}
                </Badge>
              </label>
            ))}
          </div>
        </ConfigSection>

        <ConfigSection title={t("config.cv")}>
          <FormField
            control={form.control}
            name="cv.n_folds"
            render={({ field }) => (
              <FormItem>
                <div className="flex items-center justify-between">
                  <FormLabel className="text-xs">{t("config.folds")}</FormLabel>
                  <span className="text-sm font-semibold tabular-nums">{field.value}</span>
                </div>
                <FormControl>
                  <Slider
                    min={2}
                    max={20}
                    step={1}
                    value={[field.value]}
                    onValueChange={(value) => field.onChange(Array.isArray(value) ? value[0] : value)}
                  />
                </FormControl>
              </FormItem>
            )}
          />
          <div className="grid gap-4 sm:grid-cols-3">
            <FormField
              control={form.control}
              name="cv.mode"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-xs">{t("config.cvMode")}</FormLabel>
                  <Select value={field.value} onValueChange={field.onChange}>
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      <SelectItem value="nested">{t("config.cvMode.nested")}</SelectItem>
                      <SelectItem value="single">{t("config.cvMode.single")}</SelectItem>
                    </SelectContent>
                  </Select>
                </FormItem>
              )}
            />
            {cvMode === "nested" && (
              <FormField
                control={form.control}
                name="cv.n_inner_folds"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-xs">{t("config.innerFolds")}</FormLabel>
                    <FormControl>
                      <Input
                        type="number"
                        min={2}
                        max={10}
                        value={field.value}
                        onChange={(event) => field.onChange(Number(event.target.value))}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            )}
            <FormField
              control={form.control}
              name="cv.random_state"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-xs">{t("config.seed")}</FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      value={field.value}
                      onChange={(event) => field.onChange(Number(event.target.value))}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </div>
        </ConfigSection>

        <ConfigSection title={t("config.hpo")}>
          <FormField
            control={form.control}
            name="hpo.enabled"
            render={({ field }) => (
              <FormItem className="flex items-center justify-between rounded-lg border p-3">
                <FormLabel className="text-sm font-normal">{t("config.hpo.enabled")}</FormLabel>
                <FormControl>
                  <Switch checked={field.value} onCheckedChange={field.onChange} />
                </FormControl>
              </FormItem>
            )}
          />
          {hpoEnabled && (
            <div className="grid gap-4 sm:grid-cols-2">
              <FormField
                control={form.control}
                name="hpo.n_trials"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-xs">{t("config.trials")}</FormLabel>
                    <FormControl>
                      <Input
                        type="number"
                        min={1}
                        max={500}
                        value={field.value}
                        onChange={(event) => field.onChange(Number(event.target.value))}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="hpo.timeout_s"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-xs">{t("config.timeout")}</FormLabel>
                    <FormControl>
                      <Input
                        type="number"
                        min={1}
                        placeholder={t("config.timeoutOff")}
                        value={field.value ?? ""}
                        onChange={(event) =>
                          field.onChange(event.target.value === "" ? null : Number(event.target.value))
                        }
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
          )}
        </ConfigSection>

        <ConfigSection title={t("config.explain")}>
          <FormField
            control={form.control}
            name="explain.shap"
            render={({ field }) => (
              <FormItem className="flex items-center justify-between rounded-lg border p-3">
                <FormLabel className="text-sm font-normal">{t("config.shap")}</FormLabel>
                <FormControl>
                  <Switch checked={field.value} onCheckedChange={field.onChange} />
                </FormControl>
              </FormItem>
            )}
          />
          {shapEnabled && (
            <div className="grid gap-4 sm:grid-cols-2">
              <FormField
                control={form.control}
                name="explain.background_samples"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-xs">{t("config.background")}</FormLabel>
                    <FormControl>
                      <Input
                        type="number"
                        min={10}
                        max={1000}
                        value={field.value}
                        onChange={(event) => field.onChange(Number(event.target.value))}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="explain.max_display"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-xs">{t("config.maxDisplay")}</FormLabel>
                    <FormControl>
                      <Input
                        type="number"
                        min={3}
                        max={50}
                        value={field.value}
                        onChange={(event) => field.onChange(Number(event.target.value))}
                      />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
          )}
        </ConfigSection>

        <ConfigSection title={t("config.export")}>
          <div className="grid gap-4 sm:grid-cols-3">
            <FormField
              control={form.control}
              name="exportConfig.figures_dpi"
              render={({ field }) => (
                <FormItem>
                  <FormLabel className="text-xs">{t("config.dpi")}</FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      min={72}
                      max={600}
                      step={1}
                      value={field.value}
                      onChange={(event) => field.onChange(Number(event.target.value))}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="exportConfig.include_eps"
              render={({ field }) => (
                <FormItem className="flex flex-row items-center justify-between rounded-lg border p-3">
                  <FormLabel className="text-xs font-normal">{t("config.eps")}</FormLabel>
                  <FormControl>
                    <Switch checked={field.value} onCheckedChange={field.onChange} />
                  </FormControl>
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="exportConfig.include_shap_dependence"
              render={({ field }) => (
                <FormItem className="flex flex-row items-center justify-between rounded-lg border p-3">
                  <FormLabel className="text-xs font-normal">{t("config.dependence")}</FormLabel>
                  <FormControl>
                    <Switch checked={field.value} onCheckedChange={field.onChange} />
                  </FormControl>
                </FormItem>
              )}
            />
          </div>
        </ConfigSection>
      </form>
    </Form>
  );
}

function useTrainFormInternal() {
  return useForm<TrainFormValues>({
    resolver: zodResolver(trainSchema),
    defaultValues: DEFAULT_FORM,
    mode: "onSubmit",
  });
}

export { useTrainFormInternal as useTrainForm };
