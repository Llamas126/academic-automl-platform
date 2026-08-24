"use client";

import { useMutation } from "@tanstack/react-query";
import { ArrowLeft, ArrowRight, Rocket } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { toast } from "sonner";

import {
  ConfigForm,
  useTrainForm,
  type TrainFormValues,
} from "@/components/config/config-form";
import { ReviewPanel } from "@/components/config/review-panel";
import { ProfileSummary } from "@/components/upload/profile-summary";
import { UploadPanel } from "@/components/upload/upload-panel";
import { Button } from "@/components/ui/button";
import { ApiError } from "@/lib/api/client";
import { startTraining } from "@/lib/api/endpoints";
import type { DatasetProfile } from "@/lib/api/types";
import { useI18n } from "@/lib/i18n";

const STEPS = ["wizard.step1", "wizard.step2", "wizard.step3"] as const;

export default function WizardPage() {
  const { t } = useI18n();
  const router = useRouter();
  const [step, setStep] = useState(0);
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const form = useTrainForm();

  const launchMutation = useMutation({
    mutationFn: (values: TrainFormValues) =>
      startTraining({
        dataset_id: profile!.dataset_id,
        targets: values.targets,
        models: values.models.length ? values.models : null,
        cv: values.cv,
        hpo: { ...values.hpo },
        explain: values.explain,
        export: values.exportConfig,
      }),
    onSuccess: (created) => router.push(`/jobs/${created.job_id}`),
    onError: (error: ApiError) => toast.error(t("common.error"), { description: error.detail }),
  });

  const goNext = async () => {
    if (step === 1) {
      const valid = await form.trigger();
      if (!valid) return;
    }
    setStep((current) => Math.min(current + 1, STEPS.length - 1));
  };

  return (
    <div className="space-y-6">
      <ol className="flex items-center gap-2 text-sm">
        {STEPS.map((key, index) => (
          <li key={key} className="flex items-center gap-2">
            <span
              className={`flex h-6 w-6 items-center justify-center rounded-full text-xs font-semibold ${
                index <= step ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground"
              }`}
            >
              {index + 1}
            </span>
            <span className={index === step ? "font-medium" : "text-muted-foreground"}>{t(key)}</span>
            {index < STEPS.length - 1 && <ArrowRight className="h-4 w-4 text-muted-foreground" />}
          </li>
        ))}
      </ol>

      {step === 0 && (
        <>
          <UploadPanel
            onProfiled={(next) => {
              setProfile(next);
              setStep(1);
            }}
          />
          {profile && (
            <section aria-label={t("profile.file")} className="space-y-3">
              <ProfileSummary
                profile={profile}
                onPickTarget={(name, task) => {
                  form.setValue(
                    "targets",
                    [...form.getValues("targets").filter((entry) => entry.name !== name), { name, task }],
                    { shouldValidate: true },
                  );
                  setStep(1);
                }}
              />
              <div className="flex justify-end">
                <Button onClick={() => setStep(1)}>
                  {t("wizard.next")} <ArrowRight className="ml-1 h-4 w-4" />
                </Button>
              </div>
            </section>
          )}
        </>
      )}

      {step === 1 && profile && (
        <>
          <ConfigForm profile={profile} form={form} />
          <div className="flex justify-between">
            <Button variant="outline" onClick={() => setStep(0)}>
              <ArrowLeft className="mr-1 h-4 w-4" /> {t("wizard.back")}
            </Button>
            <Button onClick={goNext}>
              {t("wizard.next")} <ArrowRight className="ml-1 h-4 w-4" />
            </Button>
          </div>
        </>
      )}

      {step === 2 && profile && (
        <>
          <ReviewPanel profile={profile} values={form.getValues()} />
          <div className="flex justify-between">
            <Button variant="outline" onClick={() => setStep(1)}>
              <ArrowLeft className="mr-1 h-4 w-4" /> {t("wizard.back")}
            </Button>
            <Button
              disabled={launchMutation.isPending}
              onClick={() =>
                launchMutation.mutate({
                  ...form.getValues(),
                  cv: { ...form.getValues("cv") },
                  hpo: { ...form.getValues("hpo") },
                  explain: form.getValues("explain"),
                  exportConfig: form.getValues("exportConfig"),
                })
              }
            >
              <Rocket className="mr-1 h-4 w-4" />
              {launchMutation.isPending ? t("common.loading") : t("wizard.launch")}
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
