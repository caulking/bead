/**
 * bead-span-label plugin
 *
 * jsPsych plugin for span selection and labeling. Supports:
 * - Token-level span selection (click, drag, shift+click)
 * - Static span display with overlapping highlights
 * - Fixed label sets and Wikidata entity search
 * - Relation annotation between spans
 * - Keyboard shortcuts for labels (1-9)
 *
 * @author Bead Project
 * @version 0.2.0
 */

import type { JsPsych, JsPsychPlugin, PluginInfo } from "../types/jspsych.js";

/** Span segment data */
interface SpanSegmentData {
  element_name: string;
  indices: number[];
}

/** Span label data */
interface SpanLabelData {
  label: string;
  label_id?: string;
}

/** Span data */
interface SpanData {
  span_id: string;
  segments: SpanSegmentData[];
  head_index?: number;
  label?: SpanLabelData;
  span_type?: string;
}

/** Relation data */
interface RelationData {
  relation_id: string;
  source_span_id: string;
  target_span_id: string;
  label?: SpanLabelData;
  directed: boolean;
}

/** Span specification from Python */
interface SpanSpecData {
  index_mode: "token" | "character";
  interaction_mode: "static" | "interactive";
  label_source: "fixed" | "wikidata";
  labels?: string[];
  label_colors?: Record<string, string>;
  allow_overlapping: boolean;
  min_spans?: number;
  max_spans?: number;
  enable_relations: boolean;
  relation_label_source: "fixed" | "wikidata";
  relation_labels?: string[];
  relation_directed: boolean;
  min_relations?: number;
  max_relations?: number;
  wikidata_language: string;
  wikidata_entity_types?: string[];
  wikidata_result_limit: number;
}

/** Display configuration */
interface SpanDisplayConfigData {
  highlight_style: "background" | "underline" | "border";
  color_palette: string[];
  show_labels: boolean;
  show_tooltips: boolean;
  token_delimiter: string;
  label_position: "inline" | "below" | "tooltip";
}

/** Bead metadata */
interface BeadMetadata {
  spans?: SpanData[];
  span_relations?: RelationData[];
  tokenized_elements?: Record<string, string[]>;
  token_space_after?: Record<string, boolean[]>;
  span_spec?: SpanSpecData;
  [key: string]: unknown;
}

/** Span event for logging */
interface SpanEvent {
  type: "select" | "deselect" | "label" | "delete" | "relation_create" | "relation_delete";
  timestamp: number;
  span_id?: string;
  relation_id?: string;
  indices?: number[];
  label?: string;
}

/** Trial parameters */
export interface SpanLabelTrialParams {
  tokens: Record<string, string[]>;
  space_after: Record<string, boolean[]>;
  spans: SpanData[];
  relations: RelationData[];
  span_spec: SpanSpecData | null;
  display_config: SpanDisplayConfigData | null;
  prompt: string;
  button_label: string;
  require_response: boolean;
  metadata: BeadMetadata;
}

/** Plugin info constant */
const info: PluginInfo = {
  name: "bead-span-label",
  parameters: {
    tokens: {
      type: 12, // OBJECT
      default: {},
    },
    space_after: {
      type: 12, // OBJECT
      default: {},
    },
    spans: {
      type: 12, // OBJECT
      default: [],
      array: true,
    },
    relations: {
      type: 12, // OBJECT
      default: [],
      array: true,
    },
    span_spec: {
      type: 12, // OBJECT
      default: null,
    },
    display_config: {
      type: 12, // OBJECT
      default: null,
    },
    prompt: {
      type: 8, // HTML_STRING
      default: "Select and label spans",
    },
    button_label: {
      type: 1, // STRING
      default: "Continue",
    },
    require_response: {
      type: 0, // BOOL
      default: true,
    },
    metadata: {
      type: 12, // OBJECT
      default: {},
    },
  },
};

const DEFAULT_PALETTE = [
  "#BBDEFB", "#C8E6C9", "#FFE0B2", "#F8BBD0",
  "#D1C4E9", "#B2EBF2", "#DCEDC8", "#FFD54F",
];

/**
 * BeadSpanLabelPlugin - jsPsych plugin for span annotation
 */
class BeadSpanLabelPlugin implements JsPsychPlugin<typeof info, SpanLabelTrialParams> {
  static info = info;

  private jsPsych: JsPsych;

  constructor(jsPsych: JsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_element: HTMLElement, trial: SpanLabelTrialParams): void {
    const start_time = performance.now();
    const events: SpanEvent[] = [];

    // Resolve config from metadata or parameters
    const tokens = Object.keys(trial.tokens).length > 0
      ? trial.tokens
      : (trial.metadata.tokenized_elements ?? {});
    const spaceAfter = Object.keys(trial.space_after).length > 0
      ? trial.space_after
      : (trial.metadata.token_space_after ?? {});
    const spanSpec = trial.span_spec ?? trial.metadata.span_spec ?? null;
    const preSpans = trial.spans.length > 0
      ? trial.spans
      : (trial.metadata.spans ?? []);
    const preRelations = trial.relations.length > 0
      ? trial.relations
      : (trial.metadata.span_relations ?? []);

    const palette = trial.display_config?.color_palette ?? DEFAULT_PALETTE;
    const isInteractive = spanSpec?.interaction_mode === "interactive";

    // Working state
    const activeSpans: SpanData[] = [...preSpans];
    const activeRelations: RelationData[] = [...preRelations];
    let selectionStart: number | null = null;
    let selectedIndices: number[] = [];
    let nextSpanId = activeSpans.length;

    // Build HTML
    let html = '<div class="bead-span-label-container">';

    if (trial.prompt) {
      html += `<div class="bead-rating-prompt">${trial.prompt}</div>`;
    }

    // Render tokens for each element
    const elementNames = Object.keys(tokens).sort();
    for (const elemName of elementNames) {
      const elemTokens = tokens[elemName] ?? [];
      const elemSpaceAfter = spaceAfter[elemName] ?? [];

      html += `<div class="bead-span-container" data-element="${elemName}">`;
      for (let i = 0; i < elemTokens.length; i++) {
        const tokenText = elemTokens[i];
        const interactive = isInteractive ? " interactive" : "";
        html += `<span class="bead-token${interactive}" data-index="${i}" data-element="${elemName}">${tokenText}</span>`;
        if (i < elemSpaceAfter.length && elemSpaceAfter[i]) {
          html += " ";
        }
      }
      html += "</div>";
    }

    // Label selector (for interactive mode with fixed labels)
    if (isInteractive && spanSpec?.labels && spanSpec.labels.length > 0) {
      html += '<div class="bead-label-selector" id="bead-label-panel" style="display:none;">';
      for (let i = 0; i < spanSpec.labels.length; i++) {
        const label = spanSpec.labels[i];
        const shortcut = i < 9 ? ` [${i + 1}]` : "";
        const color = palette[i % palette.length];
        html += `<button class="bead-label-button" data-label="${label}" style="border-left: 3px solid ${color}">${label}${shortcut}</button>`;
      }
      html += "</div>";
    }

    // Span list
    html += '<div class="bead-span-list" id="bead-span-list"></div>';

    // Relation list
    if (spanSpec?.enable_relations) {
      html += '<div class="bead-relation-list" id="bead-relation-list"></div>';
    }

    // Continue button
    html += `
      <div class="bead-rating-button-container">
        <button class="bead-button bead-continue-button" id="bead-span-continue" ${isInteractive && trial.require_response ? "disabled" : ""}>
          ${trial.button_label}
        </button>
      </div>
    `;

    html += "</div>";
    display_element.innerHTML = html;

    // Apply static span highlights
    applySpanHighlights();

    // Render span list
    renderSpanList();

    if (isInteractive) {
      setupInteractiveHandlers();
    }

    // Continue button
    const continueBtn = display_element.querySelector<HTMLButtonElement>("#bead-span-continue");
    if (continueBtn) {
      continueBtn.addEventListener("click", () => {
        endTrial();
      });
    }

    function applySpanHighlights(): void {
      // Clear existing highlights
      const allTokens = display_element.querySelectorAll<HTMLSpanElement>(".bead-token");
      for (const t of allTokens) {
        t.classList.remove("highlighted");
        t.removeAttribute("data-span-ids");
        t.removeAttribute("data-span-count");
        t.style.removeProperty("background-color");
        t.style.removeProperty("background");
      }

      // Build token -> span mapping
      const tokenSpanMap: Map<string, string[]> = new Map();
      for (const span of activeSpans) {
        for (const seg of span.segments) {
          for (const idx of seg.indices) {
            const key = `${seg.element_name}:${idx}`;
            if (!tokenSpanMap.has(key)) {
              tokenSpanMap.set(key, []);
            }
            tokenSpanMap.get(key)?.push(span.span_id);
          }
        }
      }

      // Apply highlights
      const spanColorMap = assignColors();
      for (const t of allTokens) {
        const elemName = t.getAttribute("data-element") ?? "";
        const idx = t.getAttribute("data-index") ?? "";
        const key = `${elemName}:${idx}`;
        const spanIds = tokenSpanMap.get(key) ?? [];

        if (spanIds.length > 0) {
          t.classList.add("highlighted");
          t.setAttribute("data-span-ids", spanIds.join(","));
          t.setAttribute("data-span-count", String(spanIds.length));

          if (spanIds.length === 1) {
            t.style.backgroundColor = spanColorMap.get(spanIds[0] ?? "") ?? palette[0] ?? "#BBDEFB";
          } else {
            const colors = spanIds.map(id => spanColorMap.get(id) ?? palette[0] ?? "#BBDEFB");
            t.style.background = `linear-gradient(${colors.join(", ")})`;
          }
        }
      }
    }

    function assignColors(): Map<string, string> {
      const colorMap: Map<string, string> = new Map();
      const labelColors = spanSpec?.label_colors ?? {};
      let colorIdx = 0;

      for (const span of activeSpans) {
        if (span.label?.label && labelColors[span.label.label]) {
          colorMap.set(span.span_id, labelColors[span.label.label] ?? "#BBDEFB");
        } else {
          colorMap.set(span.span_id, palette[colorIdx % palette.length] ?? "#BBDEFB");
          colorIdx++;
        }
      }
      return colorMap;
    }

    function renderSpanList(): void {
      const listEl = display_element.querySelector<HTMLDivElement>("#bead-span-list");
      if (!listEl) return;

      listEl.innerHTML = "";
      const colorMap = assignColors();

      for (const span of activeSpans) {
        const badge = document.createElement("span");
        badge.className = "bead-span-badge";
        const color = colorMap.get(span.span_id) ?? palette[0];
        badge.style.backgroundColor = color ?? "";
        const labelText = span.label?.label ?? "unlabeled";
        const spanText = getSpanText(span);
        badge.textContent = `${labelText}: "${spanText}"`;
        badge.setAttribute("data-span-id", span.span_id);

        if (isInteractive) {
          const deleteBtn = document.createElement("button");
          deleteBtn.textContent = "\u00d7";
          deleteBtn.style.cssText = "margin-left:4px;border:none;background:none;cursor:pointer;font-weight:bold;";
          deleteBtn.addEventListener("click", () => {
            deleteSpan(span.span_id);
          });
          badge.appendChild(deleteBtn);
        }

        listEl.appendChild(badge);
      }
    }

    function getSpanText(span: SpanData): string {
      const parts: string[] = [];
      for (const seg of span.segments) {
        const elemTokens = tokens[seg.element_name] ?? [];
        for (const idx of seg.indices) {
          if (idx < elemTokens.length) {
            parts.push(elemTokens[idx] ?? "");
          }
        }
      }
      return parts.join(" ");
    }

    function setupInteractiveHandlers(): void {
      const tokenEls = display_element.querySelectorAll<HTMLSpanElement>(".bead-token.interactive");

      for (const tokenEl of tokenEls) {
        tokenEl.addEventListener("click", (e) => {
          const idx = Number.parseInt(tokenEl.getAttribute("data-index") ?? "0", 10);
          const elemName = tokenEl.getAttribute("data-element") ?? "";

          if (e.shiftKey && selectionStart !== null) {
            // Range selection
            const start = Math.min(selectionStart, idx);
            const end = Math.max(selectionStart, idx);
            selectedIndices = [];
            for (let i = start; i <= end; i++) {
              selectedIndices.push(i);
            }
          } else {
            // Toggle single token
            const existingIdx = selectedIndices.indexOf(idx);
            if (existingIdx >= 0) {
              selectedIndices.splice(existingIdx, 1);
            } else {
              selectedIndices.push(idx);
            }
            selectionStart = idx;
          }

          // Update selection UI
          updateSelectionUI(elemName);

          // Show label panel if we have a selection
          const labelPanel = display_element.querySelector("#bead-label-panel");
          if (labelPanel) {
            if (selectedIndices.length > 0) {
              (labelPanel as HTMLElement).style.display = "flex";
            } else {
              (labelPanel as HTMLElement).style.display = "none";
            }
          }
        });
      }

      // Label button handlers
      const labelButtons = display_element.querySelectorAll<HTMLButtonElement>(".bead-label-button");
      for (const btn of labelButtons) {
        btn.addEventListener("click", () => {
          const label = btn.getAttribute("data-label") ?? "";
          if (selectedIndices.length > 0 && label) {
            createSpanFromSelection(label);
          }
        });
      }

      // Keyboard shortcuts for labels
      document.addEventListener("keydown", handleKeyDown);
    }

    function handleKeyDown(e: KeyboardEvent): void {
      const num = Number.parseInt(e.key, 10);
      if (!Number.isNaN(num) && num >= 1 && num <= 9) {
        const labels = spanSpec?.labels ?? [];
        if (num <= labels.length && selectedIndices.length > 0) {
          createSpanFromSelection(labels[num - 1] ?? "");
        }
      }
    }

    function updateSelectionUI(elementName: string): void {
      const tokenEls = display_element.querySelectorAll<HTMLSpanElement>(
        `.bead-token[data-element="${elementName}"]`
      );
      for (const t of tokenEls) {
        const idx = Number.parseInt(t.getAttribute("data-index") ?? "0", 10);
        if (selectedIndices.includes(idx)) {
          t.classList.add("selecting");
        } else {
          t.classList.remove("selecting");
        }
      }
    }

    function createSpanFromSelection(label: string): void {
      const elemName = elementNames[0] ?? "text";
      const spanId = `span_${nextSpanId++}`;

      const newSpan: SpanData = {
        span_id: spanId,
        segments: [{
          element_name: elemName,
          indices: [...selectedIndices].sort((a, b) => a - b),
        }],
        label: { label },
      };

      activeSpans.push(newSpan);
      events.push({
        type: "select",
        timestamp: performance.now() - start_time,
        span_id: spanId,
        indices: [...selectedIndices],
        label,
      });

      // Clear selection
      selectedIndices = [];
      selectionStart = null;

      // Update UI
      applySpanHighlights();
      renderSpanList();
      updateContinueButton();

      // Clear selection UI
      const allTokens = display_element.querySelectorAll<HTMLSpanElement>(".bead-token");
      for (const t of allTokens) {
        t.classList.remove("selecting");
      }

      // Hide label panel
      const labelPanel = display_element.querySelector("#bead-label-panel");
      if (labelPanel) {
        (labelPanel as HTMLElement).style.display = "none";
      }
    }

    function deleteSpan(spanId: string): void {
      const idx = activeSpans.findIndex(s => s.span_id === spanId);
      if (idx >= 0) {
        activeSpans.splice(idx, 1);
        events.push({
          type: "delete",
          timestamp: performance.now() - start_time,
          span_id: spanId,
        });
        applySpanHighlights();
        renderSpanList();
        updateContinueButton();
      }
    }

    function updateContinueButton(): void {
      if (!continueBtn || !isInteractive) return;
      const minSpans = spanSpec?.min_spans ?? 0;
      continueBtn.disabled = activeSpans.length < minSpans;
    }

    const endTrial = (): void => {
      // Remove keyboard listener
      document.removeEventListener("keydown", handleKeyDown);

      const trial_data: Record<string, unknown> = {
        ...trial.metadata,
        spans: activeSpans,
        relations: activeRelations,
        span_events: events,
        rt: performance.now() - start_time,
      };

      display_element.innerHTML = "";
      this.jsPsych.finishTrial(trial_data);
    };
  }
}

export { BeadSpanLabelPlugin };
