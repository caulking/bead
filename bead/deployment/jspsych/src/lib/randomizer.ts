/**
 * Trial randomization with constraint enforcement.
 *
 * Provides functions for shuffling and validating trial order
 * against ordering constraints (precedence, no-adjacent, min-distance).
 */

// Trial structure with required item_id
export interface Trial {
  item_id: string;
  [key: string]: unknown;
}

// Ordering constraint specification
export interface OrderingConstraint {
  precedence_pairs?: [string, string][];
  no_adjacent_property?: string;
  min_distance?: number;
}

// Randomizer configuration
export interface RandomizerConfig {
  hasPrecedence: boolean;
  hasNoAdjacent: boolean;
  hasMinDistance: boolean;
  hasBlocking: boolean;
  hasPractice: boolean;
  blockProperty?: string;
  practiceProperty?: string;
  randomizeWithinBlocks?: boolean;
}

// Trial metadata record
export type TrialMetadata = Record<string, Record<string, unknown>>;

// Seeded random number generator type
export type SeededRNG = () => number;

/**
 * Shuffle array in place using Fisher-Yates algorithm.
 */
export function shuffle<T>(array: T[], rng: SeededRNG): void {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const temp = array[i];
    const swapVal = array[j];
    if (temp !== undefined && swapVal !== undefined) {
      array[i] = swapVal;
      array[j] = temp;
    }
  }
}

/**
 * Get property value from nested object using dot notation.
 */
export function getPropertyValue(obj: Record<string, unknown> | undefined, path: string): unknown {
  if (obj === undefined || obj === null) {
    return undefined;
  }

  const parts = path.split(".");
  let current: unknown = obj;

  for (const part of parts) {
    if (current === undefined || current === null) {
      return undefined;
    }
    if (typeof current === "object" && current !== null) {
      current = (current as Record<string, unknown>)[part];
    } else {
      return undefined;
    }
  }

  return current;
}

/**
 * Check if trial order satisfies precedence constraints.
 * Item A must appear before item B for each pair.
 */
export function checkPrecedence(trials: Trial[], pairs: [string, string][]): boolean {
  const positions: Record<string, number> = {};
  trials.forEach((trial, idx) => {
    positions[trial.item_id] = idx;
  });

  for (const [itemA, itemB] of pairs) {
    const posA = positions[itemA];
    const posB = positions[itemB];
    if (posA !== undefined && posB !== undefined) {
      if (posA >= posB) {
        return false;
      }
    }
  }
  return true;
}

/**
 * Check if no adjacent trials have the same property value.
 */
export function checkNoAdjacent(
  trials: Trial[],
  property: string,
  metadata: TrialMetadata,
): boolean {
  for (let i = 0; i < trials.length - 1; i++) {
    const trialA = trials[i];
    const trialB = trials[i + 1];
    if (!trialA || !trialB) continue;

    const metaA = metadata[trialA.item_id];
    const metaB = metadata[trialB.item_id];

    const valueA = getPropertyValue(metaA, property);
    const valueB = getPropertyValue(metaB, property);

    if (valueA !== undefined && valueB !== undefined && valueA === valueB) {
      return false;
    }
  }
  return true;
}

/**
 * Check if minimum distance constraint is satisfied.
 * Items with the same property value must be at least minDist apart.
 */
export function checkMinDistance(
  trials: Trial[],
  property: string,
  minDist: number,
  metadata: TrialMetadata,
): boolean {
  const valuePositions: Record<string, number[]> = {};

  trials.forEach((trial, idx) => {
    const meta = metadata[trial.item_id];
    const value = getPropertyValue(meta, property);
    if (value !== undefined) {
      const key = String(value);
      if (!valuePositions[key]) {
        valuePositions[key] = [];
      }
      valuePositions[key]?.push(idx);
    }
  });

  for (const positions of Object.values(valuePositions)) {
    for (let i = 0; i < positions.length - 1; i++) {
      const posA = positions[i];
      const posB = positions[i + 1];
      if (posA !== undefined && posB !== undefined) {
        const distance = posB - posA - 1;
        if (distance < minDist) {
          return false;
        }
      }
    }
  }
  return true;
}

/**
 * Check if trial order satisfies all constraints.
 */
export function checkAllConstraints(
  trials: Trial[],
  constraints: OrderingConstraint[],
  metadata: TrialMetadata,
  config: RandomizerConfig,
): boolean {
  for (const constraint of constraints) {
    // check precedence constraints
    if (config.hasPrecedence && constraint.precedence_pairs) {
      if (constraint.precedence_pairs.length > 0) {
        if (!checkPrecedence(trials, constraint.precedence_pairs)) {
          return false;
        }
      }
    }

    // check no-adjacent constraints
    if (config.hasNoAdjacent && constraint.no_adjacent_property) {
      if (!checkNoAdjacent(trials, constraint.no_adjacent_property, metadata)) {
        return false;
      }
    }

    // check minimum distance constraints
    if (config.hasMinDistance && constraint.min_distance !== undefined) {
      if (constraint.no_adjacent_property) {
        if (
          !checkMinDistance(
            trials,
            constraint.no_adjacent_property,
            constraint.min_distance,
            metadata,
          )
        ) {
          return false;
        }
      }
    }
  }

  return true;
}

/**
 * Separate practice and main trials.
 */
export function separatePracticeTrials(
  trials: Trial[],
  metadata: TrialMetadata,
  practiceProperty: string,
): { practiceTrials: Trial[]; mainTrials: Trial[] } {
  const practiceTrials: Trial[] = [];
  const mainTrials: Trial[] = [];

  for (const trial of trials) {
    const meta = metadata[trial.item_id];
    const value = getPropertyValue(meta, practiceProperty);
    if (value === true) {
      practiceTrials.push(trial);
    } else {
      mainTrials.push(trial);
    }
  }

  return { practiceTrials, mainTrials };
}

/**
 * Group trials by block property.
 */
export function groupByBlock(
  trials: Trial[],
  metadata: TrialMetadata,
  blockProperty: string,
): Record<string, Trial[]> {
  const blocks: Record<string, Trial[]> = {};

  for (const trial of trials) {
    const meta = metadata[trial.item_id];
    const blockValue = getPropertyValue(meta, blockProperty);
    const blockKey = blockValue !== undefined ? String(blockValue) : "__undefined__";

    if (!blocks[blockKey]) {
      blocks[blockKey] = [];
    }
    blocks[blockKey]?.push(trial);
  }

  return blocks;
}

/**
 * Randomize trials with blocking.
 */
export function randomizeWithBlocking(
  mainTrials: Trial[],
  metadata: TrialMetadata,
  blockProperty: string,
  randomizeWithinBlocks: boolean,
  rng: SeededRNG,
): Trial[] {
  const blocks = groupByBlock(mainTrials, metadata, blockProperty);

  // randomize block order
  const blockKeys = Object.keys(blocks);
  shuffle(blockKeys, rng);

  let randomizedMain: Trial[] = [];
  for (const key of blockKeys) {
    const blockTrials = blocks[key];
    if (blockTrials) {
      if (randomizeWithinBlocks) {
        shuffle(blockTrials, rng);
      }
      randomizedMain = randomizedMain.concat(blockTrials);
    }
  }

  return randomizedMain;
}

/**
 * Randomize trials with rejection sampling (no blocking).
 */
export function randomizeWithRejectionSampling(
  mainTrials: Trial[],
  constraints: OrderingConstraint[],
  metadata: TrialMetadata,
  config: RandomizerConfig,
  rng: SeededRNG,
  maxAttempts = 1000,
): Trial[] {
  let randomizedMain = [...mainTrials];
  let lastAttempt = [...randomizedMain];

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    shuffle(randomizedMain, rng);
    lastAttempt = [...randomizedMain];

    if (checkAllConstraints(randomizedMain, constraints, metadata, config)) {
      break;
    }

    if (attempt === maxAttempts - 1) {
      console.warn(
        `Could not find constraint-satisfying order after ${maxAttempts} attempts. Using last attempt.`,
      );
      randomizedMain = lastAttempt;
    }
  }

  return randomizedMain;
}

/**
 * Create a randomizer function with the given configuration.
 */
export function createRandomizer(
  metadata: TrialMetadata,
  constraints: OrderingConstraint[],
  config: RandomizerConfig,
): (trials: Trial[], rng: SeededRNG) => Trial[] {
  return (trials: Trial[], rng: SeededRNG): Trial[] => {
    // separate practice trials if configured
    let practiceTrials: Trial[] = [];
    let mainTrials: Trial[];

    if (config.hasPractice && config.practiceProperty) {
      const separated = separatePracticeTrials(trials, metadata, config.practiceProperty);
      practiceTrials = separated.practiceTrials;
      mainTrials = separated.mainTrials;
    } else {
      mainTrials = [...trials];
    }

    // randomize main trials
    let randomizedMain: Trial[];

    if (config.hasBlocking && config.blockProperty) {
      randomizedMain = randomizeWithBlocking(
        mainTrials,
        metadata,
        config.blockProperty,
        config.randomizeWithinBlocks ?? true,
        rng,
      );
    } else {
      randomizedMain = randomizeWithRejectionSampling(
        mainTrials,
        constraints,
        metadata,
        config,
        rng,
      );
    }

    return practiceTrials.concat(randomizedMain);
  };
}
