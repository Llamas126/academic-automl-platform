"use client";

import { UploadCloud } from "lucide-react";
import { useRef, useState } from "react";
import { toast } from "sonner";

import { Progress } from "@/components/ui/progress";
import { Card, CardContent } from "@/components/ui/card";
import { uploadDataset } from "@/lib/api/endpoints";
import { ApiError } from "@/lib/api/client";
import type { DatasetProfile } from "@/lib/api/types";
import { MAX_UPLOAD_MB } from "@/lib/constants";
import { useI18n } from "@/lib/i18n";

const ACCEPT = ".csv,.xlsx";

export function UploadPanel({ onProfiled }: { onProfiled: (profile: DatasetProfile) => void }) {
  const { t } = useI18n();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);

  const handleFile = (file: File) => {
    const ext = file.name.slice(file.name.lastIndexOf(".")).toLowerCase();
    if (![".csv", ".xlsx"].includes(ext)) {
      toast.error(t("upload.failed"), { description: "CSV / XLSX" });
      return;
    }
    if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      toast.error(t("upload.failed"), { description: t("upload.maxSize", { size: MAX_UPLOAD_MB }) });
      return;
    }
    setProgress(0);
    uploadDataset(file, setProgress)
      .then((profile) => {
        toast.success(t("upload.success"));
        onProfiled(profile);
      })
      .catch((error: ApiError) => toast.error(t("upload.failed"), { description: error.detail }))
      .finally(() => setProgress(null));
  };

  return (
    <Card>
      <CardContent className="p-6">
        <div
          role="button"
          tabIndex={0}
          aria-label={t("upload.title")}
          onClick={() => inputRef.current?.click()}
          onKeyDown={(event) => event.key === "Enter" && inputRef.current?.click()}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            const file = event.dataTransfer.files[0];
            if (file) handleFile(file);
          }}
          className={`flex min-h-56 cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed p-8 text-center transition-colors ${
            dragging ? "border-primary bg-primary/5" : "border-muted-foreground/25 hover:border-primary/50"
          }`}
        >
          <UploadCloud className="h-10 w-10 text-muted-foreground" />
          <p className="font-medium">{t("upload.title")}</p>
          <p className="max-w-md text-sm text-muted-foreground">{t("upload.hint")}</p>
          <p className="text-xs text-muted-foreground">{t("upload.maxSize", { size: MAX_UPLOAD_MB })}</p>
          {progress !== null && (
            <div className="w-full max-w-sm space-y-1 pt-2">
              <Progress value={progress} />
              <p className="text-xs text-muted-foreground">{t("upload.uploading", { pct: progress })}</p>
            </div>
          )}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="hidden"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) handleFile(file);
            event.target.value = "";
          }}
        />
      </CardContent>
    </Card>
  );
}
