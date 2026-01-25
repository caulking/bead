/**
 * List distribution system for JATOS batch sessions.
 *
 * Manages server-side list assignment using JATOS batch sessions.
 * Supports 8 distribution strategies with strict error handling.
 */

import type { BatchSessionData, JatosAPI, JatosPromise } from "../types/jatos.js";

declare const jatos: JatosAPI;

// Strategy types
export type StrategyType =
  | "random"
  | "sequential"
  | "balanced"
  | "latin_square"
  | "stratified"
  | "weighted_random"
  | "quota_based"
  | "metadata_based";

// Distribution configuration
export interface DistributionConfig {
  strategy_type: StrategyType;
  strategy_config?: StrategyConfig;
  max_participants?: number;
  debug_mode?: boolean;
  debug_list_index?: number;
}

export interface StrategyConfig {
  factors?: string[];
  weight_expression?: string;
  normalize_weights?: boolean;
  participants_per_list?: number;
  allow_overflow?: boolean;
  filter_expression?: string;
  rank_expression?: string;
  rank_ascending?: boolean;
}

// Experiment list structure
export interface ExperimentList {
  id: string;
  name: string;
  list_number: number;
  list_metadata?: Record<string, unknown>;
  item_refs: string[];
}

// Queue entry for list assignments
interface QueueEntry {
  list_index: number;
  list_id: string;
}

// Assignment record
interface Assignment {
  list_index: number;
  list_id: string;
  assigned_at: string;
  completed: boolean;
}

// Statistics structure
interface Statistics {
  assignment_counts: Record<number, number>;
  completion_counts: Record<number, number>;
  total_assignments: number;
  total_completions: number;
}

// Item structure
export interface ExperimentItem {
  id: string;
  [key: string]: unknown;
}

/**
 * Sleep for specified milliseconds.
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Load lists from lists.jsonl file.
 */
export async function loadLists(jsonlPath: string): Promise<ExperimentList[]> {
  const response = await fetch(jsonlPath);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch lists.jsonl (HTTP ${response.status}). Expected file at: ${jsonlPath}. Verify the experiment was generated correctly using JsPsychExperimentGenerator.generate().`,
    );
  }

  const text = await response.text();
  const lists: ExperimentList[] = [];
  const lines = text.trim().split("\n");

  for (const line of lines) {
    if (line.trim()) {
      try {
        const list = JSON.parse(line) as ExperimentList;
        lists.push(list);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(
          `Failed to parse list from lists.jsonl: ${message}. ` +
            `Line content: ${line.substring(0, 100)}...`,
        );
      }
    }
  }

  if (lists.length === 0) {
    throw new Error(
      `Loaded lists.jsonl but got empty array. Verify your ExperimentLists were created and passed to generate(). File path: ${jsonlPath}`,
    );
  }

  return lists;
}

/**
 * Load items from items.jsonl file.
 */
export async function loadItems(jsonlPath: string): Promise<Record<string, ExperimentItem>> {
  const response = await fetch(jsonlPath);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch items.jsonl (HTTP ${response.status}). Expected file at: ${jsonlPath}. Verify the experiment was generated correctly.`,
    );
  }

  const text = await response.text();
  const items: Record<string, ExperimentItem> = {};
  const lines = text.trim().split("\n");

  for (const line of lines) {
    if (line.trim()) {
      try {
        const item = JSON.parse(line) as ExperimentItem;
        items[item.id] = item;
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(
          `Failed to parse item from items.jsonl: ${message}. ` +
            `Line content: ${line.substring(0, 100)}...`,
        );
      }
    }
  }

  if (Object.keys(items).length === 0) {
    throw new Error(
      `Loaded items.jsonl but got empty dictionary. Verify your Items were created and passed to generate(). File path: ${jsonlPath}`,
    );
  }

  return items;
}

/**
 * Generate balanced Latin square using Bradley's (1958) algorithm.
 */
function generateBalancedLatinSquare(n: number): number[][] {
  const square: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      if (i % 2 === 0) {
        row.push((Math.floor(i / 2) + j) % n);
      } else {
        row.push((Math.floor(i / 2) + n - j) % n);
      }
    }
    square.push(row);
  }
  return square;
}

/**
 * Fisher-Yates shuffle algorithm for array randomization.
 */
function shuffleArray<T>(array: T[]): void {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const temp = array[i];
    const swapVal = array[j];
    if (temp !== undefined && swapVal !== undefined) {
      array[i] = swapVal;
      array[j] = temp;
    }
  }
}

/**
 * Initialize queue for random strategy.
 */
function initializeRandom(
  _config: DistributionConfig,
  lists: ExperimentList[],
  maxParticipants: number,
): QueueEntry[] {
  const queue: QueueEntry[] = [];
  const perList = Math.ceil(maxParticipants / lists.length);
  for (let i = 0; i < lists.length; i++) {
    const list = lists[i];
    if (list) {
      for (let j = 0; j < perList; j++) {
        queue.push({ list_index: i, list_id: list.id });
      }
    }
  }
  shuffleArray(queue);
  return queue;
}

/**
 * Initialize queue for sequential strategy.
 */
function initializeSequential(
  _config: DistributionConfig,
  lists: ExperimentList[],
  maxParticipants: number,
): QueueEntry[] {
  const queue: QueueEntry[] = [];
  for (let i = 0; i < maxParticipants; i++) {
    const listIndex = i % lists.length;
    const list = lists[listIndex];
    if (list) {
      queue.push({ list_index: listIndex, list_id: list.id });
    }
  }
  return queue;
}

/**
 * Initialize queue and matrix for Latin square strategy.
 */
function initializeLatinSquare(
  _config: DistributionConfig,
  lists: ExperimentList[],
): { queue: QueueEntry[]; matrix: number[][] } {
  const matrix = generateBalancedLatinSquare(lists.length);
  const queue: QueueEntry[] = [];

  for (let row = 0; row < matrix.length; row++) {
    const matrixRow = matrix[row];
    if (matrixRow) {
      for (let col = 0; col < matrixRow.length; col++) {
        const listIndex = matrixRow[col];
        if (listIndex !== undefined) {
          const list = lists[listIndex];
          if (list) {
            queue.push({ list_index: listIndex, list_id: list.id });
          }
        }
      }
    }
  }

  return { queue, matrix };
}

/**
 * Initialize queue and quotas for quota-based strategy.
 */
function initializeQuotaBased(
  config: DistributionConfig,
  lists: ExperimentList[],
): { queue: QueueEntry[]; quotas: Record<number, number> } {
  const quota = config.strategy_config?.participants_per_list ?? 10;
  const quotas: Record<number, number> = {};
  const queue: QueueEntry[] = [];

  for (let i = 0; i < lists.length; i++) {
    quotas[i] = quota;
    const list = lists[i];
    if (list) {
      for (let j = 0; j < quota; j++) {
        queue.push({ list_index: i, list_id: list.id });
      }
    }
  }

  shuffleArray(queue);
  return { queue, quotas };
}

/**
 * Initialize batch session state for list distribution.
 */
async function initializeBatchSession(
  config: DistributionConfig,
  lists: ExperimentList[],
): Promise<void> {
  // set distribution config
  await jatos.batchSession.set("distribution", {
    strategy_type: config.strategy_type,
    strategy_config: config.strategy_config ?? {},
    initialized: true,
    created_at: new Date().toISOString(),
  });

  // initialize statistics
  const assignment_counts: Record<number, number> = {};
  const completion_counts: Record<number, number> = {};
  for (let i = 0; i < lists.length; i++) {
    assignment_counts[i] = 0;
    completion_counts[i] = 0;
  }

  await jatos.batchSession.set("statistics", {
    assignment_counts,
    completion_counts,
    total_assignments: 0,
    total_completions: 0,
  });

  // initialize assignments
  await jatos.batchSession.set("assignments", {});

  // strategy-specific initialization
  const maxParticipants = config.max_participants ?? 1000;

  switch (config.strategy_type) {
    case "random": {
      const randomQueue = initializeRandom(config, lists, maxParticipants);
      await jatos.batchSession.set("lists_queue", randomQueue);
      await jatos.batchSession.set("strategy_state", {});
      break;
    }

    case "sequential": {
      const seqQueue = initializeSequential(config, lists, maxParticipants);
      await jatos.batchSession.set("lists_queue", seqQueue);
      await jatos.batchSession.set("strategy_state", { next_index: 0 });
      break;
    }

    case "balanced": {
      await jatos.batchSession.set("strategy_state", {});
      break;
    }

    case "latin_square": {
      const { queue, matrix } = initializeLatinSquare(config, lists);
      await jatos.batchSession.set("lists_queue", queue);
      await jatos.batchSession.set("strategy_state", {
        latin_square_matrix: matrix,
        latin_square_position: 0,
      });
      break;
    }

    case "stratified": {
      if (!config.strategy_config?.factors || config.strategy_config.factors.length === 0) {
        throw new Error(
          `StratifiedConfig requires 'factors' in strategy_config. Got: ${JSON.stringify(config.strategy_config)}. Provide a list like ['condition', 'verb_type'].`,
        );
      }
      await jatos.batchSession.set("strategy_state", {});
      break;
    }

    case "weighted_random": {
      if (!config.strategy_config?.weight_expression) {
        throw new Error(
          `WeightedRandomConfig requires 'weight_expression' in strategy_config. Got: ${JSON.stringify(config.strategy_config)}. Provide a JavaScript expression like 'list_metadata.priority || 1.0'.`,
        );
      }
      await jatos.batchSession.set("strategy_state", {});
      break;
    }

    case "quota_based": {
      if (!config.strategy_config?.participants_per_list) {
        throw new Error(
          `QuotaConfig requires 'participants_per_list' in strategy_config. Got: ${JSON.stringify(config.strategy_config)}. Add 'participants_per_list: <int>' to your distribution_strategy config.`,
        );
      }
      const { queue: quotaQueue, quotas } = initializeQuotaBased(config, lists);
      await jatos.batchSession.set("lists_queue", quotaQueue);
      await jatos.batchSession.set("strategy_state", { remaining_quotas: quotas });
      break;
    }

    case "metadata_based": {
      const hasFilter = config.strategy_config?.filter_expression;
      const hasRank = config.strategy_config?.rank_expression;
      if (!hasFilter && !hasRank) {
        throw new Error(
          `MetadataBasedConfig requires at least one of 'filter_expression' or 'rank_expression'. Got: ${JSON.stringify(config.strategy_config)}. Add 'filter_expression' or 'rank_expression'.`,
        );
      }
      await jatos.batchSession.set("strategy_state", {});
      break;
    }

    default: {
      const _exhaustive: never = config.strategy_type;
      throw new Error(`Unknown strategy type: '${config.strategy_type}'.`);
    }
  }
}

/**
 * Atomic queue update with retry using promise callbacks.
 */
function updateQueueAtomically(
  workerId: string,
  selected: QueueEntry,
  updatedQueue: QueueEntry[],
  lists: ExperimentList[],
): Promise<number> {
  return new Promise((resolve, reject) => {
    function attemptUpdate(retries = 5): void {
      const currentQueue =
        (jatos.batchSession.get("lists_queue") as QueueEntry[] | undefined) ?? [];

      if (Math.abs(currentQueue.length - updatedQueue.length) > 1) {
        if (retries > 0) {
          setTimeout(() => attemptUpdate(retries - 1), 100 * (6 - retries));
          return;
        }
        reject(new Error("Queue modified concurrently"));
        return;
      }

      (jatos.batchSession.set("lists_queue", updatedQueue) as JatosPromise<void>)
        .then(() => {
          const assignments =
            (jatos.batchSession.get("assignments") as Record<string, Assignment>) ?? {};
          assignments[workerId] = {
            list_index: selected.list_index,
            list_id: selected.list_id,
            assigned_at: new Date().toISOString(),
            completed: false,
          };
          return jatos.batchSession.set("assignments", assignments);
        })
        .then(() => {
          const stats = (jatos.batchSession.get("statistics") as Statistics | undefined) ?? {
            assignment_counts: {},
            completion_counts: {},
            total_assignments: 0,
            total_completions: 0,
          };
          const currentCount = stats.assignment_counts[selected.list_index] ?? 0;
          stats.assignment_counts[selected.list_index] = currentCount + 1;
          stats.total_assignments += 1;
          return jatos.batchSession.set("statistics", stats);
        })
        .then(() => resolve(selected.list_index))
        .fail((error: Error) => {
          if (retries > 0) {
            setTimeout(() => attemptUpdate(retries - 1), 100 * (6 - retries));
          } else {
            reject(new Error(`Failed to update queue: ${error.message}`));
          }
        });
    }

    attemptUpdate();
  });
}

/**
 * Atomic statistics update with version checking.
 */
function updateStatisticsAtomically(
  workerId: string,
  listIndex: number,
  oldCounts: Record<number, number>,
  _oldStats: Statistics,
  lists: ExperimentList[],
): Promise<number> {
  return new Promise((resolve, reject) => {
    function attemptUpdate(retries = 5): void {
      const currentStats = (jatos.batchSession.get("statistics") as Statistics | undefined) ?? {
        assignment_counts: {},
        completion_counts: {},
        total_assignments: 0,
        total_completions: 0,
      };
      const currentCounts = currentStats.assignment_counts;

      const expectedCount = oldCounts[listIndex] ?? 0;
      const actualCount = currentCounts[listIndex] ?? 0;

      if (actualCount !== expectedCount && retries > 0) {
        setTimeout(
          () => {
            updateStatisticsAtomically(workerId, listIndex, currentCounts, currentStats, lists)
              .then(resolve)
              .catch(reject);
          },
          100 * (6 - retries),
        );
        return;
      }

      currentStats.assignment_counts[listIndex] =
        (currentStats.assignment_counts[listIndex] ?? 0) + 1;
      currentStats.total_assignments = (currentStats.total_assignments ?? 0) + 1;

      (jatos.batchSession.set("statistics", currentStats) as JatosPromise<void>)
        .then(() => {
          const assignments =
            (jatos.batchSession.get("assignments") as Record<string, Assignment>) ?? {};
          const list = lists[listIndex];
          assignments[workerId] = {
            list_index: listIndex,
            list_id: list?.id ?? "",
            assigned_at: new Date().toISOString(),
            completed: false,
          };
          return jatos.batchSession.set("assignments", assignments);
        })
        .then(() => resolve(listIndex))
        .fail((error: Error) => {
          if (retries > 0) {
            setTimeout(() => attemptUpdate(retries - 1), 100 * (6 - retries));
          } else {
            reject(new Error(`Failed to update statistics: ${error.message}`));
          }
        });
    }

    attemptUpdate();
  });
}

/**
 * Unified assignment function routing to strategy-specific implementations.
 */
async function assignList(
  workerId: string,
  config: DistributionConfig,
  lists: ExperimentList[],
): Promise<number> {
  // check existing assignment (idempotency)
  const assignments = (jatos.batchSession.get("assignments") as Record<string, Assignment>) ?? {};
  const existing = assignments[workerId];
  if (existing) {
    console.log("Worker already assigned:", existing);
    return existing.list_index;
  }

  // route to strategy-specific assignment
  switch (config.strategy_type) {
    case "random":
      return assignRandom(workerId, config, lists);
    case "sequential":
      return assignSequential(workerId, config, lists);
    case "balanced":
      return assignBalanced(workerId, config, lists);
    case "latin_square":
      return assignLatinSquare(workerId, config, lists);
    case "stratified":
      return assignStratified(workerId, config, lists);
    case "weighted_random":
      return assignWeightedRandom(workerId, config, lists);
    case "quota_based":
      return assignQuotaBased(workerId, config, lists);
    case "metadata_based":
      return assignMetadataBased(workerId, config, lists);
    default: {
      const _exhaustive: never = config.strategy_type;
      throw new Error(`Unknown strategy type: '${config.strategy_type}'.`);
    }
  }
}

/**
 * Random assignment strategy (queue-based).
 */
async function assignRandom(
  workerId: string,
  _config: DistributionConfig,
  lists: ExperimentList[],
): Promise<number> {
  const queue = (jatos.batchSession.get("lists_queue") as QueueEntry[] | undefined) ?? [];
  if (queue.length === 0) {
    throw new Error(
      "No lists available in queue for random assignment. " +
        "Verify lists.jsonl was generated and batch session initialized.",
    );
  }

  const randomIndex = Math.floor(Math.random() * queue.length);
  const selected = queue[randomIndex];
  if (!selected) {
    throw new Error("Failed to select from queue");
  }
  const updatedQueue = queue.filter((_, idx) => idx !== randomIndex);

  return updateQueueAtomically(workerId, selected, updatedQueue, lists);
}

/**
 * Sequential (round-robin) assignment strategy.
 */
async function assignSequential(
  workerId: string,
  _config: DistributionConfig,
  lists: ExperimentList[],
): Promise<number> {
  const queue = (jatos.batchSession.get("lists_queue") as QueueEntry[] | undefined) ?? [];
  const nextIndex =
    (jatos.batchSession.get("strategy_state/next_index") as number | undefined) ?? 0;

  if (nextIndex >= queue.length) {
    throw new Error(
      `Sequential queue exhausted (position ${nextIndex} >= queue length ${queue.length}). Increase max_participants or add more lists.`,
    );
  }

  const selected = queue[nextIndex];
  if (!selected) {
    throw new Error("Failed to select from queue");
  }

  // simplified: just update statistics
  const stats = (jatos.batchSession.get("statistics") as Statistics | undefined) ?? {
    assignment_counts: {},
    completion_counts: {},
    total_assignments: 0,
    total_completions: 0,
  };
  const counts = stats.assignment_counts;
  return updateStatisticsAtomically(workerId, selected.list_index, counts, stats, lists);
}

/**
 * Balanced assignment strategy (assign to least-used list).
 */
async function assignBalanced(
  workerId: string,
  _config: DistributionConfig,
  lists: ExperimentList[],
): Promise<number> {
  for (let attempt = 0; attempt < 5; attempt++) {
    const stats = (jatos.batchSession.get("statistics") as Statistics | undefined) ?? {
      assignment_counts: {},
      completion_counts: {},
      total_assignments: 0,
      total_completions: 0,
    };
    const counts = stats.assignment_counts;

    // find minimum count
    let minCount = Number.POSITIVE_INFINITY;
    const minIndices: number[] = [];
    for (let i = 0; i < lists.length; i++) {
      const count = counts[i] ?? 0;
      if (count < minCount) {
        minCount = count;
        minIndices.length = 0;
        minIndices.push(i);
      } else if (count === minCount) {
        minIndices.push(i);
      }
    }

    const selectedIndex = minIndices[Math.floor(Math.random() * minIndices.length)];
    if (selectedIndex === undefined) {
      throw new Error("No lists available");
    }

    try {
      const result = await updateStatisticsAtomically(
        workerId,
        selectedIndex,
        counts,
        stats,
        lists,
      );
      return result;
    } catch (error) {
      if (attempt === 4) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Failed to assign balanced list after 5 retries. Last error: ${message}.`);
      }
      await sleep(100 * 2 ** attempt);
    }
  }

  throw new Error("Failed to assign balanced list after retries");
}

/**
 * Latin square counterbalancing strategy.
 */
async function assignLatinSquare(
  workerId: string,
  _config: DistributionConfig,
  lists: ExperimentList[],
): Promise<number> {
  const matrix = jatos.batchSession.get("strategy_state/latin_square_matrix") as
    | number[][]
    | undefined;
  const position =
    (jatos.batchSession.get("strategy_state/latin_square_position") as number | undefined) ?? 0;

  if (!matrix || !Array.isArray(matrix) || matrix.length === 0) {
    throw new Error(
      "Latin square matrix not initialized. Verify batch session was initialized correctly.",
    );
  }

  const row = position % matrix.length;
  const matrixRow = matrix[row];
  if (!matrixRow) {
    throw new Error("Invalid matrix row");
  }
  const col = Math.floor(position / matrix.length) % matrixRow.length;
  const listIndex = matrixRow[col];
  if (listIndex === undefined) {
    throw new Error("Invalid list index from matrix");
  }

  const stats = (jatos.batchSession.get("statistics") as Statistics | undefined) ?? {
    assignment_counts: {},
    completion_counts: {},
    total_assignments: 0,
    total_completions: 0,
  };
  const counts = stats.assignment_counts;
  return updateStatisticsAtomically(workerId, listIndex, counts, stats, lists);
}

/**
 * Stratified assignment strategy (balance across factors).
 */
async function assignStratified(
  workerId: string,
  config: DistributionConfig,
  lists: ExperimentList[],
): Promise<number> {
  const factors = config.strategy_config?.factors;
  if (!factors || factors.length === 0) {
    throw new Error(
      `StratifiedConfig requires 'factors' in strategy_config. Got: ${JSON.stringify(config.strategy_config)}.`,
    );
  }

  for (let attempt = 0; attempt < 5; attempt++) {
    const stats = (jatos.batchSession.get("statistics") as Statistics | undefined) ?? {
      assignment_counts: {},
      completion_counts: {},
      total_assignments: 0,
      total_completions: 0,
    };
    const counts = stats.assignment_counts;

    // group lists by factor combinations
    const strata: Record<string, number[]> = {};
    for (let i = 0; i < lists.length; i++) {
      const list = lists[i];
      if (list) {
        const key = factors
          .map((f) => {
            const metadata = list.list_metadata as Record<string, unknown> | undefined;
            return String(metadata?.[f] ?? "null");
          })
          .join("|");
        if (!strata[key]) {
          strata[key] = [];
        }
        strata[key]?.push(i);
      }
    }

    // find stratum with minimum total assignments
    let minCount = Number.POSITIVE_INFINITY;
    let minStratumIndices: number[] = [];

    for (const [_key, indices] of Object.entries(strata)) {
      const stratumCount = indices.reduce((sum, idx) => sum + (counts[idx] ?? 0), 0);
      if (stratumCount < minCount) {
        minCount = stratumCount;
        minStratumIndices = indices;
      }
    }

    const listIndex = minStratumIndices[Math.floor(Math.random() * minStratumIndices.length)];
    if (listIndex === undefined) {
      throw new Error("No lists available in strata");
    }

    try {
      const result = await updateStatisticsAtomically(workerId, listIndex, counts, stats, lists);
      return result;
    } catch (error) {
      if (attempt === 4) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`Failed to assign stratified list after 5 retries: ${message}`);
      }
      await sleep(100 * 2 ** attempt);
    }
  }

  throw new Error("Failed to assign stratified list after retries");
}

/**
 * Weighted random assignment strategy.
 */
async function assignWeightedRandom(
  workerId: string,
  config: DistributionConfig,
  lists: ExperimentList[],
): Promise<number> {
  const expr = config.strategy_config?.weight_expression;
  if (!expr) {
    throw new Error(`WeightedRandomConfig requires 'weight_expression' in strategy_config.`);
  }

  const normalize = config.strategy_config?.normalize_weights !== false;

  // compute weights from metadata
  const weights = lists.map((list) => {
    const list_metadata = list.list_metadata ?? {};
    try {
      // biome-ignore lint/security/noGlobalEval: user-provided expression for weighted random selection
      return eval(expr) as number;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(
        `Failed to evaluate weight_expression '${expr}' for list ${list.name}: ${message}.`,
      );
    }
  });

  // normalize if requested
  let w = weights;
  if (normalize) {
    const sum = weights.reduce((a, b) => a + b, 0);
    if (sum === 0) {
      throw new Error("Sum of weights is 0. Cannot normalize.");
    }
    w = weights.map((weight) => weight / sum);
  }

  // sample from cumulative distribution
  const cumulative: number[] = [];
  let sum = 0;
  for (const weight of w) {
    sum += weight;
    cumulative.push(sum);
  }

  const lastCumulative = cumulative[cumulative.length - 1];
  const random = Math.random() * (lastCumulative ?? 1);
  let listIndex = lists.length - 1;
  for (let i = 0; i < cumulative.length; i++) {
    const cumulativeValue = cumulative[i];
    if (cumulativeValue !== undefined && random <= cumulativeValue) {
      listIndex = i;
      break;
    }
  }

  const stats = (jatos.batchSession.get("statistics") as Statistics | undefined) ?? {
    assignment_counts: {},
    completion_counts: {},
    total_assignments: 0,
    total_completions: 0,
  };
  const counts = stats.assignment_counts;
  return updateStatisticsAtomically(workerId, listIndex, counts, stats, lists);
}

/**
 * Quota-based assignment strategy.
 */
async function assignQuotaBased(
  workerId: string,
  config: DistributionConfig,
  lists: ExperimentList[],
): Promise<number> {
  if (!config.strategy_config?.participants_per_list) {
    throw new Error(`QuotaConfig requires 'participants_per_list' in strategy_config.`);
  }

  const quotas =
    (jatos.batchSession.get("strategy_state/remaining_quotas") as
      | Record<number, number>
      | undefined) ?? {};

  // find available lists
  const available: number[] = [];
  for (let i = 0; i < lists.length; i++) {
    const quota = quotas[i];
    if (quota !== undefined && quota > 0) {
      available.push(i);
    }
  }

  if (available.length === 0) {
    if (config.strategy_config.allow_overflow === true) {
      return assignBalanced(workerId, config, lists);
    }
    throw new Error(
      `All lists have reached quota and allow_overflow=false. Current quotas: ${JSON.stringify(quotas)}.`,
    );
  }

  const listIndex = available[Math.floor(Math.random() * available.length)];
  if (listIndex === undefined) {
    throw new Error("No lists available");
  }

  const stats = (jatos.batchSession.get("statistics") as Statistics | undefined) ?? {
    assignment_counts: {},
    completion_counts: {},
    total_assignments: 0,
    total_completions: 0,
  };
  const counts = stats.assignment_counts;
  return updateStatisticsAtomically(workerId, listIndex, counts, stats, lists);
}

/**
 * Metadata-based assignment strategy.
 */
async function assignMetadataBased(
  workerId: string,
  config: DistributionConfig,
  lists: ExperimentList[],
): Promise<number> {
  const hasFilter = config.strategy_config?.filter_expression;
  const hasRank = config.strategy_config?.rank_expression;

  if (!hasFilter && !hasRank) {
    throw new Error(
      `MetadataBasedConfig requires at least one of 'filter_expression' or 'rank_expression'.`,
    );
  }

  // filter lists
  let available = lists.map((list, idx) => ({ list, idx, score: 0 }));

  if (hasFilter) {
    const filterExpr = config.strategy_config?.filter_expression;
    available = available.filter((item) => {
      const list_metadata = item.list.list_metadata ?? {};
      try {
        // biome-ignore lint/security/noGlobalEval: user-provided filter expression
        return eval(filterExpr as string) as boolean;
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(
          `Failed to evaluate filter_expression '${filterExpr}' for list ${item.list.name}: ${message}.`,
        );
      }
    });

    if (available.length === 0) {
      throw new Error(
        `No lists match filter_expression: '${filterExpr}'. ` +
          `All ${lists.length} lists were filtered out.`,
      );
    }
  }

  // rank lists
  if (hasRank) {
    const rankExpr = config.strategy_config?.rank_expression;
    const ascending = config.strategy_config?.rank_ascending !== false;

    available = available.map((item) => {
      const list_metadata = item.list.list_metadata ?? {};
      let score: number;
      try {
        // biome-ignore lint/security/noGlobalEval: user-provided rank expression
        score = eval(rankExpr as string) as number;
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(
          `Failed to evaluate rank_expression '${rankExpr}' for list ${item.list.name}: ${message}.`,
        );
      }
      return { ...item, score };
    });

    available.sort((a, b) => (ascending ? a.score - b.score : b.score - a.score));
  }

  const first = available[0];
  if (!first) {
    throw new Error("No lists available after filtering");
  }
  const listIndex = first.idx;

  const stats = (jatos.batchSession.get("statistics") as Statistics | undefined) ?? {
    assignment_counts: {},
    completion_counts: {},
    total_assignments: 0,
    total_completions: 0,
  };
  const counts = stats.assignment_counts;
  return updateStatisticsAtomically(workerId, listIndex, counts, stats, lists);
}

/**
 * Mark participant as completed.
 */
async function markCompletedInternal(workerId: string): Promise<void> {
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const allAssignments =
        (jatos.batchSession.get("assignments") as Record<string, Assignment>) ?? {};
      const assignment = allAssignments[workerId];

      if (!assignment) {
        console.warn("No assignment found for worker:", workerId);
        return;
      }

      if (assignment.completed) {
        console.log("Worker already marked as completed:", workerId);
        return;
      }

      assignment.completed = true;
      allAssignments[workerId] = assignment;
      await jatos.batchSession.set("assignments", allAssignments);

      const stats = (jatos.batchSession.get("statistics") as Statistics | undefined) ?? {
        assignment_counts: {},
        completion_counts: {},
        total_assignments: 0,
        total_completions: 0,
      };
      const currentCount = stats.completion_counts[assignment.list_index] ?? 0;
      stats.completion_counts[assignment.list_index] = currentCount + 1;
      stats.total_completions = (stats.total_completions ?? 0) + 1;
      await jatos.batchSession.set("statistics", stats);

      return;
    } catch (error) {
      if (attempt === 4) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(
          `Failed to mark worker ${workerId} as completed after 5 retries: ${message}.`,
        );
      }
      await sleep(100 * 2 ** attempt);
    }
  }
}

/**
 * ListDistributor class for managing list distribution.
 */
export class ListDistributor {
  private config: DistributionConfig;
  private lists: ExperimentList[];
  private workerId: string | null = null;
  private assignedListIndex: number | null = null;

  constructor(config: DistributionConfig, lists: ExperimentList[]) {
    if (!config) {
      throw new Error(
        "ListDistributor requires config parameter. " +
          "Pass the distribution_strategy from your config.",
      );
    }

    if (!lists || lists.length === 0) {
      throw new Error(
        "ListDistributor requires non-empty lists array. " +
          "Verify lists.jsonl was loaded correctly.",
      );
    }

    this.config = config;
    this.lists = lists;
  }

  /**
   * Initialize distributor and assign list to current worker.
   */
  async initialize(): Promise<number> {
    if (!this.lists || this.lists.length === 0) {
      throw new Error(
        "Cannot initialize: no lists available. " +
          "Verify lists.jsonl was loaded correctly and contains at least one list.",
      );
    }

    this.workerId = jatos.workerId;

    if (!this.workerId) {
      throw new Error(
        "JATOS workerId not available. " +
          "This experiment requires JATOS. " +
          "Ensure you are running this through JATOS, not as a standalone file.",
      );
    }

    if (!this.config.strategy_type) {
      throw new Error(
        "Invalid distribution config: missing strategy_type. " +
          "Verify distribution.json was loaded correctly.",
      );
    }

    try {
      await this._initializeBatchSession();
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Failed to initialize batch session: ${message}.`);
    }

    // debug mode: always return same list
    if (this.config.debug_mode) {
      const debugIndex = this.config.debug_list_index ?? 0;
      if (debugIndex < 0 || debugIndex >= this.lists.length) {
        throw new Error(
          `Invalid debug_list_index: ${debugIndex}. ` +
            `Must be between 0 and ${this.lists.length - 1}.`,
        );
      }
      console.log("Debug mode: assigning list", debugIndex);
      this.assignedListIndex = debugIndex;
      return this.assignedListIndex;
    }

    try {
      this.assignedListIndex = await assignList(this.workerId, this.config, this.lists);
      console.log(`Assigned worker ${this.workerId} to list ${this.assignedListIndex}`);
      return this.assignedListIndex;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(
        `Failed to assign list: ${message}. ` +
          `Worker ID: ${this.workerId}, Strategy: ${this.config.strategy_type}.`,
      );
    }
  }

  /**
   * Get the assigned list object.
   */
  getAssignedList(): ExperimentList {
    if (this.assignedListIndex === null) {
      throw new Error("List not yet assigned. Call initialize() first.");
    }

    if (this.assignedListIndex >= this.lists.length) {
      throw new Error(
        `Assigned list index ${this.assignedListIndex} out of bounds. ` +
          `Only ${this.lists.length} lists available.`,
      );
    }

    const list = this.lists[this.assignedListIndex];
    if (!list) {
      throw new Error(`List at index ${this.assignedListIndex} is undefined.`);
    }
    return list;
  }

  /**
   * Mark current participant as completed.
   */
  async markCompleted(): Promise<void> {
    if (this.workerId === null || this.assignedListIndex === null) {
      console.warn("Cannot mark completed: not initialized");
      return;
    }

    try {
      await markCompletedInternal(this.workerId);
      console.log(`Marked worker ${this.workerId} as completed`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`Failed to mark worker ${this.workerId} as completed: ${message}.`);
    }
  }

  /**
   * Get current distribution statistics.
   */
  getStatistics(): Statistics | undefined {
    return jatos.batchSession.get("statistics") as Statistics | undefined;
  }

  /**
   * Initialize batch session with lock mechanism.
   */
  private async _initializeBatchSession(): Promise<void> {
    if (jatos.batchSession.defined("distribution/initialized")) {
      console.log("Batch session already initialized");
      return;
    }

    console.log("Initializing batch session...");

    const lockAcquired = await this._acquireLock("init_lock");

    if (!lockAcquired) {
      await this._waitForInitialization();
      return;
    }

    try {
      if (jatos.batchSession.defined("distribution/initialized")) {
        return;
      }

      await initializeBatchSession(this.config, this.lists);
      console.log("Batch session initialized");
    } finally {
      await this._releaseLock("init_lock");
    }
  }

  /**
   * Acquire initialization lock.
   */
  private async _acquireLock(lockName: string, timeout = 5000): Promise<boolean> {
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      try {
        const lockValue = jatos.batchSession.get(lockName);

        if (!lockValue) {
          await jatos.batchSession.set(lockName, {
            holder: this.workerId,
            acquired_at: new Date().toISOString(),
          });
          return true;
        }

        await sleep(100);
      } catch (error) {
        console.warn("Error acquiring lock:", error);
        await sleep(100);
      }
    }

    console.warn(`Failed to acquire lock '${lockName}' within ${timeout}ms`);
    return false;
  }

  /**
   * Release initialization lock.
   */
  private async _releaseLock(lockName: string): Promise<void> {
    try {
      await jatos.batchSession.remove(lockName);
    } catch (error) {
      console.error("Error releasing lock:", error);
    }
  }

  /**
   * Wait for initialization to complete.
   */
  private async _waitForInitialization(timeout = 10000): Promise<void> {
    const startTime = Date.now();

    while (Date.now() - startTime < timeout) {
      if (jatos.batchSession.defined("distribution/initialized")) {
        return;
      }
      await sleep(200);
    }

    throw new Error(
      `Batch session initialization timeout (${timeout}ms). This may indicate network issues or JATOS server problems.`,
    );
  }
}
