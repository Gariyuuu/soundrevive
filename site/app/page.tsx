import type { ReactNode } from "react";
import summary from "./data/summary.json";

const BAR_HEIGHTS = [0.38, 0.62, 1, 0.7, 0.46];
const ACCENT_INDEX = 2;

const TIER_ORDER = ["LIGHT", "MODERATE", "SEVERE", "EXTREME"] as const;
const METHOD_ORDER = [
  "passthrough",
  "spectral_gate",
  "wiener",
  "median_declick",
  "stft_mask_denoiser",
] as const;
const METHOD_LABEL: Record<string, string> = {
  passthrough: "Passthrough",
  spectral_gate: "Spectral gate",
  wiener: "Wiener",
  median_declick: "Median declick",
  stft_mask_denoiser: "ML denoiser",
};

type TierRow = { tier: string; method: string; si_sdr_db: number; log_spectral_distance_db: number };
const rows = summary.tier_method_summary as TierRow[];

function siSdr(tier: string, method: string): number | undefined {
  return rows.find((r) => r.tier === tier && r.method === method)?.si_sdr_db;
}

export default function Home() {
  const { standard, headline_finding: headline } = summary;

  return (
    <main className="mx-auto w-full max-w-3xl flex-1 px-6 py-16 sm:py-20">
      <header className="flex items-center gap-3">
        <div className="flex h-8 w-8 shrink-0 items-end justify-center gap-[2px] rounded-md bg-surface p-1.5">
          {BAR_HEIGHTS.map((h, i) => (
            <div
              key={i}
              className="w-[3px] rounded-[1px]"
              style={{
                height: `${Math.round(h * 100)}%`,
                background: i === ACCENT_INDEX ? "var(--accent)" : "var(--accent-dim)",
              }}
            />
          ))}
        </div>
        <span className="font-mono text-sm tracking-wide text-muted">SoundRevive</span>
      </header>

      <h1 className="mt-10 text-3xl font-medium leading-tight sm:text-4xl">
        Restoration can make historical audio sound cleaner.
        <br />
        <span className="text-muted">Did it stay closer to the original, or drift from it?</span>
      </h1>

      <p className="mt-6 max-w-xl text-sm leading-relaxed text-muted">
        A research benchmark for ML restoration of degraded historical audio, built around one
        question: signal-fidelity metrics and content preservation don&rsquo;t automatically move
        together. This page reports what a first real test found &mdash; every number below is
        generated from a checked-in result file, never hand-typed.
      </p>

      <section className="mt-14 grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Eval speakers" value={String(standard.n_eval_pool_speakers)} />
        <Stat label="Held-out clips" value={String(standard.n_eval_pool_clips)} />
        <Stat label="Degradation tiers" value={String(standard.n_tiers)} />
        <Stat label="Methods compared" value={String(standard.n_methods)} />
      </section>

      <section className="mt-16">
        <SectionLabel>Headline finding &mdash; STANDARD tier, speaker-held-out</SectionLabel>
        <p className="mt-3 text-sm leading-relaxed text-foreground/90">
          The discriminative ML denoiser was trained on 6 speakers and evaluated on 6 it had
          never heard. Checked per speaker, at the two most severe degradation tiers:
        </p>
        <div className="mt-6 grid gap-4 sm:grid-cols-2">
          <FindingCard
            tone="accent"
            metric={`${Math.round(headline.ml_beats_wiener_on_si_sdr_severe_extreme_fraction * 100)}%`}
            label="of held-out speaker&times;tier cases: ML denoiser beats classical Wiener filtering on SI-SDR (signal fidelity)"
          />
          <FindingCard
            tone="warn"
            metric={`${Math.round(headline.ml_worse_wer_delta_than_wiener_severe_extreme_fraction * 100)}%`}
            label="of those same cases: the ML denoiser also increases word-error rate more than Wiener does &mdash; it&rsquo;s less faithful to the words actually said"
          />
        </div>
        <p className="mt-4 text-xs leading-relaxed text-muted">
          N = {headline.n_speaker_tier_combos_checked} held-out speaker&times;tier combinations.
          Real and consistent across every held-out speaker tried, but too small a sample for a
          bootstrapped effect size &mdash; see{" "}
          <code className="text-muted">research/claims_registry.md</code> row C6.
        </p>
      </section>

      <section className="mt-16">
        <SectionLabel>Signal fidelity by tier (SI-SDR, dB, higher is better)</SectionLabel>
        <div className="mt-4 overflow-x-auto rounded-lg border border-border">
          <table className="w-full min-w-[520px] text-sm">
            <thead>
              <tr className="border-b border-border text-left text-muted">
                <th className="px-4 py-3 font-normal">Tier</th>
                {METHOD_ORDER.map((m) => (
                  <th key={m} className="px-4 py-3 font-normal">
                    {METHOD_LABEL[m]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {TIER_ORDER.map((tier) => (
                <tr key={tier} className="border-b border-border last:border-0">
                  <td className="px-4 py-3 font-mono text-xs text-muted">{tier}</td>
                  {METHOD_ORDER.map((method) => {
                    const v = siSdr(tier, method);
                    const isMl = method === "stft_mask_denoiser";
                    return (
                      <td
                        key={method}
                        className="px-4 py-3 tabular-nums"
                        style={{ color: isMl ? "var(--accent)" : undefined }}
                      >
                        {v !== undefined ? v.toFixed(1) : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-muted">
          Held-out eval_pool speakers only (n = {standard.n_eval_pool_clips} clips). Source:{" "}
          <code className="text-muted">results/standard/benchmark.parquet</code>.
        </p>
      </section>

      <section className="mt-16 rounded-lg border border-border bg-surface p-5 text-sm text-muted">
        <SectionLabel>What this site does not yet cover</SectionLabel>
        <ul className="mt-3 list-inside list-disc space-y-1.5">
          <li>No generative-restoration method exists yet &mdash; only classical baselines and one discriminative ML denoiser.</li>
          <li>No human listening-study data exists anywhere in this project.</li>
          <li>No fidelity-vs-quality frontier, figures, or paper yet.</li>
        </ul>
        <p className="mt-4">
          Full status:{" "}
          <a className="text-accent underline" href="https://github.com/Gariyuuu/soundrevive/blob/main/docs/HANDOFF.md">
            docs/HANDOFF.md
          </a>
          . Source: <a className="text-accent underline" href="https://github.com/Gariyuuu/soundrevive">github.com/Gariyuuu/soundrevive</a>.
        </p>
      </section>

      <footer className="mt-16 text-xs text-muted">
        Data: LibriSpeech dev-clean (CC BY 4.0) &middot; LibriVox (CC0). See{" "}
        <code>data/manifest.json</code> for full provenance.
      </footer>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface px-4 py-3">
      <div className="text-2xl font-medium tabular-nums">{value}</div>
      <div className="mt-1 text-xs text-muted">{label}</div>
    </div>
  );
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <h2 className="font-mono text-xs uppercase tracking-wider text-muted">{children}</h2>
  );
}

function FindingCard({
  metric,
  label,
  tone,
}: {
  metric: string;
  label: string;
  tone: "accent" | "warn";
}) {
  return (
    <div className="rounded-lg border border-border bg-surface p-5">
      <div
        className="text-3xl font-medium tabular-nums"
        style={{ color: tone === "accent" ? "var(--accent)" : "var(--accent-dim)" }}
      >
        {metric}
      </div>
      <p className="mt-2 text-xs leading-relaxed text-muted">{label}</p>
    </div>
  );
}
