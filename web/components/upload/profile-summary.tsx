"use client";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { ColumnRole, DatasetProfile } from "@/lib/api/types";
import { useI18n } from "@/lib/i18n";

const ROLE_VARIANT: Record<ColumnRole, "default" | "secondary" | "outline" | "destructive"> = {
  numeric: "default",
  categorical: "secondary",
  boolean: "outline",
  datetime: "outline",
  identifier: "destructive",
  text_free: "destructive",
};

export function ProfileSummary({
  profile,
  onPickTarget,
}: {
  profile: DatasetProfile;
  onPickTarget?: (name: string, task: "classification" | "regression") => void;
}) {
  const { t } = useI18n();

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-3">
        <Card>
          <CardHeader className="pb-1">
            <CardTitle className="text-xs font-medium text-muted-foreground">{t("profile.file")}</CardTitle>
          </CardHeader>
          <CardContent className="truncate text-sm font-semibold">{profile.filename}</CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-1">
            <CardTitle className="text-xs font-medium text-muted-foreground">{t("profile.rows")}</CardTitle>
          </CardHeader>
          <CardContent className="text-2xl font-bold">{profile.n_rows.toLocaleString()}</CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-1">
            <CardTitle className="text-xs font-medium text-muted-foreground">{t("profile.columns")}</CardTitle>
          </CardHeader>
          <CardContent className="text-2xl font-bold">{profile.n_cols}</CardContent>
        </Card>
      </div>

      {profile.target_suggestions.length > 0 && onPickTarget && (
        <div className="space-y-1.5">
          <p className="text-sm font-medium">{t("profile.suggestions")}</p>
          <div className="flex flex-wrap gap-2">
            {profile.target_suggestions.map((suggestion) => (
              <Badge
                key={suggestion.name}
                variant="outline"
                title={suggestion.rationale}
                className="cursor-pointer hover:bg-accent"
                onClick={() => onPickTarget(suggestion.name, suggestion.suggested_task)}
              >
                {suggestion.name} ·{" "}
                {t(suggestion.suggested_task === "classification" ? "config.task.classification" : "config.task.regression")}
              </Badge>
            ))}
          </div>
        </div>
      )}

      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>{t("profile.colName")}</TableHead>
                <TableHead>{t("profile.colRole")}</TableHead>
                <TableHead className="w-40">{t("profile.missing")}</TableHead>
                <TableHead className="text-right">{t("profile.unique")}</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {profile.columns.map((column) => (
                <TableRow key={column.name}>
                  <TableCell className="font-mono text-sm">
                    {column.name}
                    {column.excluded_from_features && (
                      <span className="ml-2 text-xs text-muted-foreground">({t("profile.excluded")})</span>
                    )}
                  </TableCell>
                  <TableCell>
                    <Badge variant={ROLE_VARIANT[column.role]}>{column.role}</Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-muted">
                        <div
                          className="h-full rounded-full bg-destructive/70"
                          style={{ width: `${Math.min(100, column.missing_pct)}%` }}
                        />
                      </div>
                      <span className="w-10 text-right text-xs tabular-nums text-muted-foreground">
                        {column.missing_pct.toFixed(1)}%
                      </span>
                    </div>
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{column.n_unique}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
