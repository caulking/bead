/**
 * Type declarations for JATOS (Just Another Tool for Online Studies) API.
 *
 * JATOS provides a server-side infrastructure for running online experiments
 * with features like batch sessions, worker management, and data storage.
 */

/** JATOS URL query parameters (typically from Prolific/MTurk) */
export interface JatosUrlParams {
  /** Prolific participant ID */
  PROLIFIC_PID?: string;
  /** Prolific study ID */
  STUDY_ID?: string;
  /** Prolific session ID */
  SESSION_ID?: string;
  /** Generic participant ID */
  participant_id?: string;
  /** Any other URL parameters */
  [key: string]: string | undefined;
}

/** Batch session data stored in JATOS */
export interface BatchSessionData {
  /** Assignment queue for random/sequential distribution */
  assignment_queue?: number[];
  /** Counts of participants assigned to each list */
  assignment_counts?: Record<string, number>;
  /** Latin square counter for counterbalancing */
  latin_square_counter?: number;
  /** Quota tracking per list */
  quotas?: Record<string, number>;
  /** Workers currently assigned (for completion tracking) */
  active_workers?: Record<string, number>;
  /** Workers who completed the study */
  completed_workers?: string[];
  /** Custom metadata for stratification */
  stratification_state?: Record<string, unknown>;
  /** Version for optimistic locking */
  version?: number;
  /** Any other batch session data */
  [key: string]: unknown;
}

/** JATOS batch session API */
export interface JatosBatchSession {
  /**
   * Get a value from the batch session by path.
   * Uses dot notation (e.g., "assignment_counts.list_0").
   */
  get<T = unknown>(path: string): T | undefined;

  /**
   * Check if a path is defined in the batch session.
   */
  defined(path: string): boolean;

  /**
   * Set a value in the batch session.
   * Returns a promise that resolves when the write is confirmed.
   */
  set<T>(path: string, value: T): JatosPromise<void>;

  /**
   * Replace the entire batch session data.
   * Useful for atomic updates.
   */
  replace(data: BatchSessionData): JatosPromise<void>;

  /**
   * Remove a value from the batch session.
   */
  remove(path: string): JatosPromise<void>;

  /**
   * Find a value using JSON pointer or path.
   */
  find(path: string): unknown;

  /**
   * Get all batch session data.
   */
  getAll(): BatchSessionData;

  /**
   * Clear all batch session data.
   */
  clear(): JatosPromise<void>;
}

/** JATOS-style promise with .fail() for error handling */
export interface JatosPromise<T> extends Promise<T> {
  /**
   * Handle errors (JATOS uses .fail() instead of .catch()).
   */
  fail(onRejected: (reason: Error) => void): JatosPromise<T>;

  /**
   * Chain operations (alias for .then()).
   */
  then<TResult1 = T, TResult2 = never>(
    onfulfilled?: ((value: T) => TResult1 | PromiseLike<TResult1>) | null,
    onrejected?: ((reason: unknown) => TResult2 | PromiseLike<TResult2>) | null,
  ): JatosPromise<TResult1 | TResult2>;
}

/** Component result state */
export type ComponentResultState =
  | "PRE"
  | "STARTED"
  | "DATA_RETRIEVED"
  | "RESULTDATA_POSTED"
  | "FINISHED"
  | "RELOADED"
  | "ABORTED"
  | "FAIL";

/** Study result state */
export type StudyResultState =
  | "PRE"
  | "STARTED"
  | "DATA_RETRIEVED"
  | "FINISHED"
  | "ABORTED"
  | "FAIL";

/** Main JATOS API interface */
export interface JatosAPI {
  /** Current JATOS version */
  version: string;

  /** Worker ID assigned by JATOS */
  workerId: string;

  /** Current study result ID */
  studyResultId: string;

  /** Current component result ID */
  componentResultId: string;

  /** Study ID */
  studyId: string;

  /** Component ID */
  componentId: string;

  /** Batch ID */
  batchId: string;

  /** Group result ID (if in group study) */
  groupResultId?: string;

  /** URL query parameters */
  urlQueryParameters: JatosUrlParams;

  /** Study properties set in JATOS GUI */
  studyProperties: Record<string, unknown>;

  /** Component properties set in JATOS GUI */
  componentProperties: Record<string, unknown>;

  /** Study session data (persists across components) */
  studySessionData: Record<string, unknown>;

  /** Batch session API */
  batchSession: JatosBatchSession;

  /**
   * Called when JATOS is ready.
   */
  onLoad(callback: () => void): void;

  /**
   * Submit result data to JATOS.
   */
  submitResultData(data: string | object): JatosPromise<void>;

  /**
   * Append to result data.
   */
  appendResultData(data: string | object): JatosPromise<void>;

  /**
   * End the study successfully.
   */
  endStudy(successful?: boolean, message?: string): void;

  /**
   * End the study and redirect to a URL.
   */
  endStudyAndRedirect(url: string, message?: string): void;

  /**
   * End the current component and move to next.
   */
  startNextComponent(): void;

  /**
   * End the current component and move to a specific component.
   */
  startComponent(componentId: number | string): void;

  /**
   * End the current component and move to the last component.
   */
  startLastComponent(): void;

  /**
   * Abort the study.
   */
  abortStudy(message?: string): void;

  /**
   * Add an abort button to the page.
   */
  addAbortButton(): void;

  /**
   * Remove the abort button.
   */
  removeAbortButton(): void;

  /**
   * Log a message to the JATOS console.
   */
  log(message: string): void;

  /**
   * Set study session data.
   */
  setStudySessionData(data: Record<string, unknown>): JatosPromise<void>;

  /**
   * Upload a file to JATOS.
   */
  uploadResultFile(file: File | Blob, filename: string): JatosPromise<{ url: string }>;

  /**
   * Download study assets.
   */
  downloadAssetFile(filename: string): Promise<Blob>;

  /**
   * Get the component result state.
   */
  getComponentResultState(): ComponentResultState;

  /**
   * Get the study result state.
   */
  getStudyResultState(): StudyResultState;
}

/** Global JATOS instance */
declare global {
  const jatos: JatosAPI;
}
