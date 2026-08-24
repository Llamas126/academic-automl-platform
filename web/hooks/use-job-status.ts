"use client";

import { useQuery } from "@tanstack/react-query";

import { getJobStatus } from "@/lib/api/endpoints";
import type { JobState } from "@/lib/api/types";

const TERMINAL: JobState[] = ["SUCCESS", "FAILURE"];

export function useJobStatus(jobId: string) {
  return useQuery({
    queryKey: ["job", jobId],
    queryFn: ({ signal }) => getJobStatus(jobId, signal),
    refetchInterval: (query) => {
      const state = query.state.data?.state;
      return state && TERMINAL.includes(state) ? false : 2_000;
    },
    refetchIntervalInBackground: true,
    enabled: Boolean(jobId),
  });
}
