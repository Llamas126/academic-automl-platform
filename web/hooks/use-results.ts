"use client";

import { useQuery } from "@tanstack/react-query";

import { getFigures, getResults } from "@/lib/api/endpoints";

export function useResults(jobId: string) {
  return useQuery({
    queryKey: ["results", jobId],
    queryFn: () => getResults(jobId),
    enabled: Boolean(jobId),
    staleTime: Infinity,
  });
}

export function useFigures(jobId: string) {
  return useQuery({
    queryKey: ["figures", jobId],
    queryFn: () => getFigures(jobId),
    enabled: Boolean(jobId),
    staleTime: Infinity,
  });
}
