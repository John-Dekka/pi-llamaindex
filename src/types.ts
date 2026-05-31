/**
 * pi-llamaindex — Type definitions
 */

export interface Frontmatter {
	title?: string;
	category?: string;
	tags?: string[];
	[key: string]: unknown;
}

export interface IndexState {
	indexedPaths: string[];
	indexedAt: string | null;
	fileCount: number;
	chunkCount: number;
	tags: string[];
	/** Embedding model used to build this index (for stale-detection on model switch). */
	embedModel?: string;
}

export interface QueryResult {
	text: string;
	score: number;
	file: string;
	fileName: string;
	title?: string;
	category?: string;
	tags?: string;
	description?: string;
}

/**
 * Cached LlamaIndex modules.
 * Uses minimal interfaces covering only the parts we consume,
 * since LlamaIndex types are unstable and dynamically imported.
 */
export interface LiModules {
	llamaindex: typeof import("llamaindex");
	huggingface?: {
		HuggingFaceEmbedding: new (config: Record<string, unknown>) => unknown;
	};
	openai?: {
		OpenAIEmbedding: new (config: Record<string, unknown>) => unknown;
	};
}

/**
 * Cached cross-encoder reranker model.
 */
export interface RerankerModel {
	tokenizer: {
		(input: string | string[], options?: Record<string, unknown>): Promise<Record<string, unknown>>;
		(input: string[], options?: Record<string, unknown>): Promise<Record<string, unknown>>;
	};
	model: {
		(input: Record<string, unknown>): Promise<{ logits: { dims: number[]; data: Float32Array } }>;
	};
	softmax: (arr: Float32Array | number[]) => Float32Array;
}

/**
 * Minimal interface for LlamaIndex Document objects.
 */
export interface LlamaIndexDocument {
	text: string;
	metadata: Record<string, unknown>;
}

/**
 * Minimal interface for a LlamaIndex vector store index.
 * Only covers methods we actually use.
 */
export interface LlamaIndexIndex {
	insert(doc: LlamaIndexDocument): Promise<void>;
	asRetriever(config?: { similarityTopK?: number }): LlamaIndexRetriever;
}

/**
 * Minimal interface for a LlamaIndex retriever.
 */
export interface LlamaIndexRetriever {
	retrieve(params: { query: string }): Promise<Array<{ node: { metadata: Record<string, unknown>; getContent(mode: unknown): string }; score: number }>>;
}
