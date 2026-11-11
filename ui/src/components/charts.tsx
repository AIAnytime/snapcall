import { useMemo } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Json } from "../api";
import { Card } from "./ui";

export type Series = { key: string; split: string; label: string; color: string };

function ema(values: (number | null)[], alpha: number) {
  if (alpha <= 0) return values;
  let last: number | null = null;
  return values.map((v) => {
    if (v === null || v === undefined || !isFinite(v)) return null;
    last = last === null ? v : last * alpha + v * (1 - alpha);
    return last;
  });
}

function fmtTick(v: number) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1000) return `${(v / 1000).toFixed(1)}k`;
  if (a < 0.001) return v.toExponential(0);
  if (a < 1) return v.toFixed(3).replace(/0+$/, "");
  return v.toFixed(2).replace(/\.?0+$/, "");
}

export function MetricChart({ title, rows, series, smoothing = 0, height = 180, log = false }: { title: string; rows: Json[]; series: Series[]; smoothing?: number; height?: number; log?: boolean }) {
  const data = useMemo(() => {
    const steps = new Map<number, Json>();
    for (const s of series) {
      const pts = rows.filter((r) => r.split === s.split && typeof r[s.key] === "number");
      const sm = ema(pts.map((p) => p[s.key]), s.split === "val" ? 0 : smoothing);
      pts.forEach((p, i) => {
        const row = steps.get(p.step) || { step: p.step };
        row[`${s.split}:${s.key}`] = sm[i];
        row[`${s.split}:${s.key}:raw`] = p[s.key];
        steps.set(p.step, row);
      });
    }
    return [...steps.values()].sort((a, b) => a.step - b.step);
  }, [rows, series, smoothing]);
  const has = data.length > 0;
  const last = series.map((s) => {
    const v = [...data].reverse().find((d) => typeof d[`${s.split}:${s.key}:raw`] === "number");
    return v ? v[`${s.split}:${s.key}:raw`] : undefined;
  });
  return (
    <Card className="p-3.5">
      <div className="mb-2 flex items-start justify-between gap-2">
        <div className="text-[12.5px] font-medium">{title}</div>
        <div className="flex flex-wrap justify-end gap-x-3 gap-y-0.5">
          {series.map((s, i) => (
            <span key={s.split + s.key} className="flex items-center gap-1.5 text-[11.5px] text-muted">
              <span className="h-0.5 w-2.5 rounded" style={{ background: s.color }} />
              {s.label}
              {last[i] !== undefined && <span className="font-mono text-fg tabular">{fmtTick(last[i])}</span>}
            </span>
          ))}
        </div>
      </div>
      <div style={{ height }}>
        {has ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 4, right: 6, bottom: 0, left: -12 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="step" type="number" domain={["dataMin", "dataMax"]} tick={{ fill: "var(--faint)", fontSize: 11 }} tickLine={false} axisLine={{ stroke: "var(--border)" }} />
              <YAxis scale={log ? "log" : "auto"} domain={["auto", "auto"]} tickFormatter={fmtTick} tick={{ fill: "var(--faint)", fontSize: 11 }} tickLine={false} axisLine={false} width={52} allowDataOverflow />
              <Tooltip
                contentStyle={{ background: "var(--panel)", border: "1px solid var(--border-strong)", borderRadius: 6, fontSize: 12 }}
                labelStyle={{ color: "var(--muted)" }}
                labelFormatter={(l) => `step ${l}`}
                formatter={(v: any, n: any) => [typeof v === "number" ? v.toPrecision(4) : v, series.find((s) => `${s.split}:${s.key}` === n)?.label ?? n]}
              />
              {series.map((s) => (
                <Line key={s.split + s.key} dataKey={`${s.split}:${s.key}`} stroke={s.color} strokeWidth={1.6} dot={s.split === "val" ? { r: 2.5, strokeWidth: 0, fill: s.color } : false} connectNulls isAnimationActive={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center text-[12px] text-faint">Waiting for data…</div>
        )}
      </div>
    </Card>
  );
}

export function Reliability({ bins }: { bins: { lo: number; hi: number; count: number; confidence: number; accuracy: number }[] }) {
  const W = 260, H = 220, P = 30;
  const x = (v: number) => P + v * (W - P - 8);
  const y = (v: number) => H - P - v * (H - P - 8);
  const max = Math.max(1, ...bins.map((b) => b.count));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-[360px]">
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={t}>
          <line x1={x(0)} x2={x(1)} y1={y(t)} y2={y(t)} stroke="var(--grid)" />
          <text x={P - 6} y={y(t) + 3} fontSize="9" textAnchor="end" fill="var(--faint)">{t}</text>
          <text x={x(t)} y={H - P + 13} fontSize="9" textAnchor="middle" fill="var(--faint)">{t}</text>
        </g>
      ))}
      {bins.map((b, i) =>
        b.count ? (
          <g key={i}>
            <rect x={x(b.lo) + 1} width={x(b.hi) - x(b.lo) - 2} y={y(b.accuracy)} height={y(0) - y(b.accuracy)} fill="var(--accent)" opacity={0.25 + 0.6 * (b.count / max)} rx="1.5">
              <title>{`conf ${b.lo.toFixed(2)}–${b.hi.toFixed(2)} · acc ${b.accuracy.toFixed(3)} · n=${b.count}`}</title>
            </rect>
            {Math.abs(b.accuracy - b.confidence) > 0.02 && (
              <rect x={x(b.lo) + 1} width={x(b.hi) - x(b.lo) - 2} y={Math.min(y(b.accuracy), y(b.confidence))} height={Math.abs(y(b.accuracy) - y(b.confidence))} fill="var(--danger)" opacity={0.35} />
            )}
          </g>
        ) : null,
      )}
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="var(--muted)" strokeDasharray="3 3" />
      <text x={(x(0) + x(1)) / 2} y={H - 3} fontSize="10" textAnchor="middle" fill="var(--muted)">confidence</text>
      <text x={9} y={(y(0) + y(1)) / 2} fontSize="10" textAnchor="middle" fill="var(--muted)" transform={`rotate(-90 9 ${(y(0) + y(1)) / 2})`}>accuracy</text>
    </svg>
  );
}

export function Confusion({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const max = Math.max(1, ...matrix.flat());
  const n = labels.length;
  const cell = n > 20 ? 14 : n > 10 ? 22 : 34;
  return (
    <div className="overflow-auto">
      <table className="border-separate" style={{ borderSpacing: 2 }}>
        <thead>
          <tr>
            <th />
            {labels.map((l) => (
              <th key={l} className="h-24 align-bottom" style={{ width: cell }}>
                <div className="w-3 origin-bottom-left translate-x-3 -rotate-60 whitespace-nowrap text-left text-[10.5px] font-normal text-muted">{n > 30 ? "" : l}</div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="pr-2 text-right text-[11px] text-muted whitespace-nowrap max-w-[160px] truncate">{labels[i]}</td>
              {row.map((v, j) => (
                <td key={j} title={`true ${labels[i]} → pred ${labels[j]}: ${v}`} className="rounded-[3px] text-center font-mono text-[10px]"
                  style={{ width: cell, height: cell, background: v ? (i === j ? `color-mix(in srgb, var(--accent) ${15 + 85 * (v / max)}%, transparent)` : `color-mix(in srgb, var(--danger) ${20 + 80 * (v / max)}%, transparent)`) : "var(--panel-2)", color: v / max > 0.5 ? "var(--accent-fg)" : "var(--muted)" }}>
                  {cell >= 22 && v ? v : ""}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-2 text-[11.5px] text-faint">Rows: true label · columns: predicted</div>
    </div>
  );
}

export function ProbBars({ probs, top }: { probs: Record<string, number>; top?: string }) {
  const items = Object.entries(probs).sort((a, b) => b[1] - a[1]).slice(0, 12);
  return (
    <div className="space-y-1.5">
      {items.map(([k, v]) => (
        <div key={k} className="grid grid-cols-[minmax(80px,180px)_1fr_56px] items-center gap-3 text-[12.5px]">
          <div className={k === top ? "truncate font-medium" : "truncate text-muted"}>{k}</div>
          <div className="h-2 rounded-full bg-panel-2">
            <div className="h-2 rounded-full transition-all" style={{ width: `${v * 100}%`, background: k === top ? "var(--accent)" : "var(--border-strong)" }} />
          </div>
          <div className="text-right font-mono tabular text-muted">{(v * 100).toFixed(1)}%</div>
        </div>
      ))}
    </div>
  );
}
