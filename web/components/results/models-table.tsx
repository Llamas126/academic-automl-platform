"use client";

import { ChevronDown, ChevronRight } from "lucide-react";
import { Fragment, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from "recharts";

import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { ModelResult } from "@/lib/api/types";
import { useI18n } from "@/lib/i18n";

function FoldsSparkline({ values }: { values: number[] }) {
  const width = 120;
  const height = 28;
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const points = values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * width;
      const y = height - ((value - min) / span) * (height - 4) - 2;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg width={width} height={height} className="overflow-visible">
      <polyline points={points} fill="none" stroke="currentColor" strokeWidth={1.5} className="text-primary" />
    </svg>
  );
}

export function ModelsTable({ models }: { models: ModelResult[] }) {
  const { t } = useI18n();
  const [expanded, setExpanded] = useState<string | null>(null);
  const sorted = [...models].sort((a, b) => a.rank - b.rank);

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-12">{t("results.rank")}</TableHead>
          <TableHead>{t("results.model")}</TableHead>
          <TableHead>{t("results.metric")}</TableHead>
          <TableHead>{t("results.ci95")}</TableHead>
          <TableHead className="w-10" />
        </TableRow>
      </TableHeader>
      <TableBody>
        {sorted.map((model) => {
          const primary = model.metrics.find((metric) => metric.metric === model.primary_metric) ?? model.metrics[0];
          const open = expanded === model.model;
          return (
            <Fragment key={model.model}>
              <TableRow
                className={`cursor-pointer ${open ? "bg-accent/40" : ""}`}
                onClick={() => setExpanded(open ? null : model.model)}
              >
                <TableCell>
                  {model.rank === 1 ? (
                    <Badge>#{model.rank}</Badge>
                  ) : (
                    <span className="tabular-nums text-muted-foreground">#{model.rank}</span>
                  )}
                </TableCell>
                <TableCell className="font-medium">
                  {open ? <ChevronDown className="mr-1 inline h-3.5 w-3.5" /> : <ChevronRight className="mr-1 inline h-3.5 w-3.5" />}
                  {model.model}
                </TableCell>
                <TableCell className="tabular-nums">
                  {primary.mean.toFixed(4)} ± {primary.std.toFixed(4)}
                  <span className="ml-1 text-xs text-muted-foreground">{primary.metric}</span>
                </TableCell>
                <TableCell className="tabular-nums text-muted-foreground">
                  [{primary.ci95_low.toFixed(3)}, {primary.ci95_high.toFixed(3)}]
                </TableCell>
                <TableCell />
              </TableRow>
              {open && (
                <TableRow>
                  <TableCell colSpan={5} className="bg-muted/30 p-4">
                    <div className="space-y-2">
                      <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                        {t("results.allMetrics")}
                      </p>
                      {model.metrics.map((metric) => (
                        <div
                          key={metric.metric}
                          className="flex items-center justify-between gap-4 rounded-md border bg-background px-3 py-2"
                        >
                          <div className="min-w-0">
                            <span className="font-mono text-xs">{metric.metric}</span>
                            <span className="ml-2 text-sm tabular-nums">
                              {metric.mean.toFixed(4)} ± {metric.std.toFixed(4)}
                            </span>
                            <span className="ml-2 text-xs tabular-nums text-muted-foreground">
                              [{metric.ci95_low.toFixed(3)}, {metric.ci95_high.toFixed(3)}]
                            </span>
                          </div>
                          <div className="text-primary">
                            <FoldsSparkline values={metric.folds} />
                          </div>
                        </div>
                      ))}
                      {Object.keys(model.best_params).length > 0 && (
                        <pre className="max-h-32 overflow-auto rounded-md border bg-background p-2 font-mono text-[11px]">
                          {JSON.stringify(model.best_params, null, 2)}
                        </pre>
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              )}
            </Fragment>
          );
        })}
      </TableBody>
    </Table>
  );
}

export function ComparisonChart({ models }: { models: ModelResult[] }) {
  const sorted = [...models].sort((a, b) => a.primary_mean - b.primary_mean);
  const data = sorted.map((model) => ({
    name: model.model.replace(/ (Regressor|Classifier|Regression)$/, ""),
    value: model.primary_mean,
    rank: model.rank,
  }));
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16 }}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" domain={["auto", "auto"]} tick={{ fontSize: 11 }} />
          <YAxis type="category" dataKey="name" width={130} tick={{ fontSize: 11 }} />
          <RechartsTooltip formatter={(value) => Number(value).toFixed(4)} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((entry, index) => (
              <Cell
                key={entry.name}
                fill={entry.rank === 1 ? "var(--primary)" : `color-mix(in oklab, var(--primary) ${25 + index * 15}%, transparent)`}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
