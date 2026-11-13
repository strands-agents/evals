// The schedule on the candidate axis, drawn once: the candidate numbers; one row of brackets per level of rounds (as
// recorded: each scope a schedule opened), numbered within its level; which blocks each candidate changed; and which
// candidates measured nothing. Every row is labeled; nothing here is inferred from the candidates' values.

import type { Level, Scope, StudyModel } from "../model/study";
import { blockColor, INK_2, MARK, MUTED } from "../encoding/theme";
import { useHover } from "../interaction/hover";
import { useGo } from "../router";
import { textWidth, type CandidateAxis } from "./axis";

const TICKS = 18;
const LEVEL = 24;
const ROW = 15;

export function levelLabel(level: Level): string {
  return level.outer ? `${level.outer} rounds` : "rounds";
}

/** Whether the change rows say anything: not when every candidate changed every block. */
export function changesShown(study: StudyModel): boolean {
  return study.candidates.some((t) => t.role !== "baseline" && study.changed(t).length < study.blockOrder.length);
}

export function statusShown(study: StudyModel): boolean {
  return study.candidates.some((t) => study.markKind(t) === "unmeasured");
}

/** Every label the track draws on its left, for sizing the shared margin. */
export function scheduleLabels(study: StudyModel): string[] {
  return [...study.levels.map(levelLabel), ...(changesShown(study) ? study.blockOrder : []), "not measured"];
}

function Bracket({ study, scope, axis, y }: { study: StudyModel; scope: Scope; axis: CandidateAxis; y: number }) {
  const first = scope.candidates[0];
  if (first === undefined) return null;
  const done = scope.candidates[scope.candidates.length - 1];
  // the record no longer marks a round open or closed; the last round of a running run is the one still going
  const open = study.state === "running" && done === (study.candidates.at(-1)?.number ?? -1);
  const x0 = axis.x(first) - axis.step / 2 + 1.5;
  const xDone = axis.x(done) + axis.step / 2 - 1.5;
  const number = study.roundNumber(scope);
  const fits = number.length * 6.5 + 6 < xDone - x0;
  const title = [
    `${scope.outer ?? ""} round ${number}`.trim(),
    `candidates ${first}–${done}`,
    open ? "running" : "",
  ].filter(Boolean).join(" · ");
  return (
    <g>
      <title>{title}</title>
      <path d={`M${x0},${y + 6}V${y}H${xDone}`} fill="none" stroke={MARK} strokeWidth={1.2} />
      {open ? (
        <path d={`M${xDone},${y}H${xDone + 8}`} fill="none" stroke={MUTED} strokeWidth={1.2} strokeDasharray="1.5 3" />
      ) : (
        <path d={`M${xDone},${y}V${y + 6}`} fill="none" stroke={MARK} strokeWidth={1.2} />
      )}
      {fits && (
        <text x={x0 + 3} y={y - 4} className="track-num">
          {number}
        </text>
      )}
    </g>
  );
}

export function scheduleHeight(study: StudyModel): number {
  return TICKS + study.levels.length * LEVEL + (changesShown(study) ? study.blockOrder.length * ROW + 6 : 0) + (statusShown(study) ? ROW + 4 : 0) + 4;
}

export function ScheduleTrack({ study, axis, width }: { study: StudyModel; axis: CandidateAxis; width: number }) {
  const { hover, set } = useHover();
  const go = useGo();
  const height = scheduleHeight(study);
  const levelTop = TICKS;
  const changeTop = levelTop + study.levels.length * LEVEL + 4;
  const showChanges = changesShown(study);
  const statusTop = changeTop + (showChanges ? study.blockOrder.length * ROW + 2 : 0);
  const left = axis.left - 10;
  const tickW = Math.max(2, Math.min(8, axis.step * 0.6));
  return (
    <svg width={width} height={height} className="track" onMouseLeave={() => set(null)}>
      {hover.n !== null && <rect x={axis.x(hover.n) - axis.step / 2} y={0} width={axis.step} height={height} className="hover-col" />}
      {axis.ticks.map((n) => (
        <text key={n} x={axis.x(n)} y={12} textAnchor="middle" className="tick">
          {n}
        </text>
      ))}
      {study.levels.map((level, i) => {
        const y = levelTop + i * LEVEL + 16;
        return (
          <g key={`${level.depth}-${level.node}`}>
            <text x={left} y={y + 3} textAnchor="end" className="track-label">
              {levelLabel(level)}
            </text>
            {level.scopes.map((scope) => (
              <Bracket key={scope.id} study={study} scope={scope} axis={axis} y={y} />
            ))}
          </g>
        );
      })}
      {showChanges &&
        study.blockOrder.map((b, i) => {
          const cy = changeTop + i * ROW + ROW / 2;
          return (
            <g key={b}>
              <text x={left} y={cy} dy="0.32em" textAnchor="end" className="track-label">
                {b}
              </text>
              <rect x={left - textWidth(b) - 13} y={cy - 4} width={8} height={8} rx={2} fill={blockColor(study, b)} />
              <line x1={axis.left} x2={axis.right} y1={cy} y2={cy} className="track-rule" />
              {study.candidates.map((t) =>
                study.changed(t).includes(b) ? <rect key={t.number} x={axis.x(t.number) - tickW / 2} y={cy - 4} width={tickW} height={8} rx={Math.min(2, tickW / 2)} fill={blockColor(study, b)} /> : null,
              )}
            </g>
          );
        })}
      {statusShown(study) && (
        <g>
          <text x={left} y={statusTop + ROW / 2} dy="0.32em" textAnchor="end" className="track-label">
            not measured
          </text>
          {study.candidates.map((t) => {
            const cx = axis.x(t.number);
            const cy = statusTop + ROW / 2;
            if (study.markKind(t) === "unmeasured")
              return <path key={t.number} d={`M${cx - 3},${cy - 3}L${cx + 3},${cy + 3}M${cx - 3},${cy + 3}L${cx + 3},${cy - 3}`} stroke={INK_2} strokeWidth={1.5} />;
            return null;
          })}
        </g>
      )}
      {study.candidates.map((t) => (
        <rect
          key={t.number}
          x={axis.x(t.number) - axis.step / 2}
          y={0}
          width={axis.step}
          height={height}
          fill="transparent"
          style={{ cursor: "pointer" }}
          onMouseMove={(e) => set(t.number, e)}
          onClick={() => go(study.id, { page: "candidate", n: t.number })}
        />
      ))}
    </svg>
  );
}
