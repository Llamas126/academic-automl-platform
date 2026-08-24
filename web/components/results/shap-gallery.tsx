"use client";

import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { figureUrl } from "@/lib/api/endpoints";
import { useI18n } from "@/lib/i18n";

export function ShapGallery({
  jobId,
  figures,
  loading,
}: {
  jobId: string;
  figures: string[];
  loading?: boolean;
}) {
  const { t } = useI18n();

  if (loading) {
    return (
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {[0, 1, 2].map((index) => (
          <Skeleton key={index} className="aspect-[4/3] rounded-lg" />
        ))}
      </div>
    );
  }

  if (figures.length === 0) {
    return <p className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">{t("results.shapEmpty")}</p>;
  }

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">{t("results.shapHint")}</p>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {figures.map((figure) => (
          <Dialog key={figure}>
            <DialogTrigger
              render={
                <button
                  type="button"
                  className="group overflow-hidden rounded-lg border bg-background transition-shadow hover:shadow-md"
                  aria-label={figure}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={figureUrl(jobId, figure)}
                    alt={figure}
                    loading="lazy"
                    className="aspect-[4/3] w-full object-contain p-1 transition-transform group-hover:scale-[1.02]"
                  />
                  <p className="truncate px-2 py-1 text-left text-[11px] text-muted-foreground">
                    {figure.split("/").pop()}
                  </p>
                </button>
              }
            />
            <DialogContent className="max-w-3xl">
              <DialogHeader>
                <DialogTitle className="font-mono text-xs">{figure}</DialogTitle>
              </DialogHeader>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={figureUrl(jobId, figure)}
                alt={figure}
                className="max-h-[75vh] w-full object-contain"
              />
            </DialogContent>
          </Dialog>
        ))}
      </div>
    </div>
  );
}
