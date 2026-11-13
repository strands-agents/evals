// The one legend: the mark kinds this study has, then the reference lines a view draws, then a view's own items.
// Glyphs are drawn with the charts' own components and colors, so the legend cannot drift from the marks.

import type { ReactNode, SVGProps } from "react";
import { MarkGlyph } from "../charts/kit";
import { INK, LIMIT, MARK } from "../encoding/theme";
import type { StudyModel } from "../model/study";

export function LegendItem({ glyph, text }: { glyph: ReactNode; text: string }) {
  return (
    <span className="lg">
      {glyph}
      {text}
    </span>
  );
}

export function Glyph({ children }: { children: ReactNode }) {
  return (
    <svg width={22} height={18} aria-hidden>
      {children}
    </svg>
  );
}

export function LineGlyph(props: SVGProps<SVGLineElement>) {
  return (
    <Glyph>
      <line x1={2} x2={20} y1={9} y2={9} {...props} />
    </Glyph>
  );
}

export function Legend({ study, interval, baseline = true, bestLine, limit, children }: {
  study: StudyModel;
  interval?: boolean;
  baseline?: boolean;
  bestLine?: boolean;
  limit?: boolean;
  children?: ReactNode;
}) {
  const kinds = new Set(study.candidates.map((t) => study.markKind(t)));
  return (
    <div className="legend-bar">
      <LegendItem glyph={<MarkGlyph kind="ok" />} text={study.constraints.length ? "feasible" : "candidate"} />
      {study.constraints.length > 0 && <LegendItem glyph={<MarkGlyph kind="infeasible" />} text="infeasible" />}
      {kinds.has("ineligible") && <LegendItem glyph={<MarkGlyph kind="ineligible" />} text="not comparable" />}
      <LegendItem glyph={<MarkGlyph kind="ok" best />} text={study.bestIsSet ? "Pareto set" : "best"} />
      {interval && (
        <LegendItem
          glyph={
            <Glyph>
              <line x1={11} x2={11} y1={3} y2={15} stroke={MARK} strokeOpacity={0.3} strokeWidth={1.2} />
            </Glyph>
          }
          text="95% interval"
        />
      )}
      {baseline && study.baseline && <LegendItem glyph={<LineGlyph stroke={MARK} strokeOpacity={0.55} strokeDasharray="4 4" />} text="baseline" />}
      {bestLine && !study.bestIsSet && <LegendItem glyph={<LineGlyph stroke={INK} strokeOpacity={0.6} strokeWidth={1.5} />} text="best so far" />}
      {limit && <LegendItem glyph={<LineGlyph stroke={LIMIT} strokeWidth={1.5} />} text="limit" />}
      {children}
    </div>
  );
}
