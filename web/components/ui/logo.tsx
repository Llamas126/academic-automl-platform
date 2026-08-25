import Image from "next/image";

import { cn } from "@/lib/utils";

type LogoVariant = "isotype" | "full";

interface LogoProps {
  variant?: LogoVariant;
  className?: string;
}

export function Logo({ variant = "isotype", className }: LogoProps) {
  if (variant === "full") {
    return (
      <span className={cn("flex items-center gap-2", className)}>
        <Image
          src="/isotype-logo.svg"
          alt="Academic AutoML Platform"
          width={28}
          height={28}
          unoptimized
          className="h-7 w-7"
        />
        <span className="hidden text-sm font-semibold sm:inline">
          Academic AutoML Platform
        </span>
      </span>
    );
  }

  return (
    <Image
      src="/isotype-logo.svg"
      alt="Academic AutoML Platform"
      width={28}
      height={28}
      unoptimized
      className={cn("h-7 w-7", className)}
    />
  );
}
