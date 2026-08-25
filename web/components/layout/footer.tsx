import Image from "next/image";

export function Footer() {
  return (
    <footer className="border-t py-4 text-center text-xs text-muted-foreground">
      <div className="mx-auto flex max-w-6xl items-center justify-center gap-2 px-4">
        <Image
          src="/isotype-logo.svg"
          alt=""
          width={16}
          height={16}
          unoptimized
          className="h-4 w-4 opacity-50 grayscale"
          aria-hidden="true"
        />
        <span>Academic AutoML Platform · FastAPI + Next.js</span>
      </div>
    </footer>
  );
}
