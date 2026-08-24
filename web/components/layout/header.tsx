"use client";

import { Globe, Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useI18n, type Lang } from "@/lib/i18n";

const LANGS: { value: Lang; label: string; flag: string }[] = [
  { value: "es", label: "Español", flag: "🇪🇸" },
  { value: "en", label: "English", flag: "🇬🇧" },
];

function ThemeToggle() {
  const { setTheme, resolvedTheme } = useTheme();
  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label="Toggle theme"
      onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
    >
      <Sun className="h-4 w-4 dark:hidden" />
      <Moon className="hidden h-4 w-4 dark:block" />
    </Button>
  );
}

export function Header() {
  const { t, lang, setLang } = useI18n();

  return (
    <header className="sticky top-0 z-40 border-b bg-background/80 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-4 px-4">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <span className="flex h-7 w-7 items-center justify-center rounded-md bg-primary text-primary-foreground text-sm font-bold">
            ML
          </span>
          <span className="hidden sm:inline">{t("app.title")}</span>
        </Link>
        <nav className="ml-auto flex items-center gap-1 text-sm">
          <Button variant="ghost" size="sm" render={<Link href="/" />}>
            {t("nav.train")}
          </Button>
          <Button variant="ghost" size="sm" render={<Link href="/predict" />}>
            {t("nav.predict")}
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger
              render={
                <Button variant="ghost" size="sm" className="gap-1.5">
                  <Globe className="h-4 w-4" />
                  <span className="uppercase">{lang}</span>
                </Button>
              }
            />
            <DropdownMenuContent align="end">
              {LANGS.map((item) => (
                <DropdownMenuItem
                  key={item.value}
                  onClick={() => setLang(item.value)}
                  className={lang === item.value ? "bg-accent" : ""}
                >
                  <span>{item.flag}</span> {item.label}
                </DropdownMenuItem>
              ))}
            </DropdownMenuContent>
          </DropdownMenu>
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}
