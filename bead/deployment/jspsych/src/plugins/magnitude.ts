/**
 * bead-magnitude plugin
 *
 * jsPsych plugin for numeric magnitude input (bounded or unbounded).
 *
 * Follows the same metadata-spreading pattern as all other bead plugins.
 *
 * @author Bead Project
 * @version 0.2.0
 */

import type { JsPsych, JsPsychPlugin, PluginInfo } from "../types/jspsych.js";

/** Bead item/template metadata */
interface BeadMetadata {
  [key: string]: unknown;
}

/** Magnitude trial parameters */
export interface MagnitudeTrialParams {
  /** The prompt/question to display */
  prompt: string;
  /** HTML stimulus to display */
  stimulus: string;
  /** Unit label displayed next to the input */
  unit: string;
  /** Minimum allowed value (null = unbounded) */
  input_min: number | null;
  /** Maximum allowed value (null = unbounded) */
  input_max: number | null;
  /** Step size for the input */
  step: number | null;
  /** Placeholder text for the input */
  placeholder: string;
  /** Whether to require a response */
  require_response: boolean;
  /** Text for the continue button */
  button_label: string;
  /** Complete item and template metadata */
  metadata: BeadMetadata;
}

/** Plugin info constant */
const info: PluginInfo = {
  name: "bead-magnitude",
  parameters: {
    prompt: {
      type: 8, // ParameterType.HTML_STRING
      default: "Enter a value:",
    },
    stimulus: {
      type: 8, // ParameterType.HTML_STRING
      default: "",
    },
    unit: {
      type: 1, // ParameterType.STRING
      default: "",
    },
    input_min: {
      type: 3, // ParameterType.FLOAT
      default: null,
    },
    input_max: {
      type: 3, // ParameterType.FLOAT
      default: null,
    },
    step: {
      type: 3, // ParameterType.FLOAT
      default: null,
    },
    placeholder: {
      type: 1, // ParameterType.STRING
      default: "",
    },
    require_response: {
      type: 0, // ParameterType.BOOL
      default: true,
    },
    button_label: {
      type: 1, // ParameterType.STRING
      default: "Continue",
    },
    metadata: {
      type: 12, // ParameterType.OBJECT
      default: {},
    },
  },
};

/**
 * BeadMagnitudePlugin - jsPsych plugin for numeric magnitude input
 */
class BeadMagnitudePlugin implements JsPsychPlugin<typeof info, MagnitudeTrialParams> {
  static info = info;

  private jsPsych: JsPsych;

  constructor(jsPsych: JsPsych) {
    this.jsPsych = jsPsych;
  }

  trial(display_element: HTMLElement, trial: MagnitudeTrialParams): void {
    const start_time = performance.now();

    // Build HTML
    let html = '<div class="bead-magnitude-container">';

    if (trial.prompt) {
      html += `<div class="bead-magnitude-prompt">${trial.prompt}</div>`;
    }

    if (trial.stimulus) {
      html += `<div class="bead-magnitude-stimulus">${trial.stimulus}</div>`;
    }

    html += '<div class="bead-magnitude-input-wrapper">';
    html += '<input type="number" class="bead-magnitude-input" id="bead-magnitude-input"';
    if (trial.input_min !== null) html += ` min="${trial.input_min}"`;
    if (trial.input_max !== null) html += ` max="${trial.input_max}"`;
    if (trial.step !== null) html += ` step="${trial.step}"`;
    if (trial.placeholder) html += ` placeholder="${trial.placeholder}"`;
    html += ">";
    if (trial.unit) {
      html += `<span class="bead-magnitude-unit">${trial.unit}</span>`;
    }
    html += "</div>";

    // Continue button
    const disabled = trial.require_response ? "disabled" : "";
    html += `
      <div class="bead-magnitude-button-container">
        <button class="bead-button bead-continue-button" id="bead-magnitude-continue" ${disabled}>
          ${trial.button_label}
        </button>
      </div>
    `;

    html += "</div>";

    display_element.innerHTML = html;

    // Input listener
    const input = display_element.querySelector<HTMLInputElement>("#bead-magnitude-input");
    const continueBtn = display_element.querySelector<HTMLButtonElement>("#bead-magnitude-continue");

    if (input) {
      input.addEventListener("input", () => {
        if (continueBtn) {
          continueBtn.disabled = trial.require_response && input.value.trim() === "";
        }
      });
      input.focus();
    }

    if (continueBtn) {
      continueBtn.addEventListener("click", () => {
        if (!trial.require_response || (input && input.value.trim() !== "")) {
          end_trial();
        }
      });
    }

    const end_trial = (): void => {
      const rt = performance.now() - start_time;
      const value = input ? Number.parseFloat(input.value) : null;

      const trial_data: Record<string, unknown> = {
        ...trial.metadata,
        response: Number.isNaN(value ?? Number.NaN) ? null : value,
        rt: rt,
      };

      display_element.innerHTML = "";
      this.jsPsych.finishTrial(trial_data);
    };
  }
}

export { BeadMagnitudePlugin };
