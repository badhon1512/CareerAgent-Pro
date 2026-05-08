"use client";

import { FormEvent, useRef, useState } from "react";
import {
  Bot,
  BriefcaseBusiness,
  Download,
  ExternalLink,
  FileText,
  LinkIcon,
  Send,
  UserRound,
} from "lucide-react";

type AgentToolCall = {
  name: string;
  input: Record<string, unknown>;
  output: Record<string, unknown>;
};

type AgentTrace = {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  estimated_cost_usd: number;
  tool_call_count: number;
  tool_call_time_ms: number;
  total_time_ms: number;
  model: string | null;
};

type AgentResponse = {
  result: string;
  tool_calls: AgentToolCall[];
  trace?: AgentTrace | null;
};

type CVExtractResponse = {
  filename: string;
  content_type: string | null;
  text: string;
  character_count: number;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  links?: string[];
  toolCalls?: AgentToolCall[];
  trace?: AgentTrace | null;
};

type JobResult = {
  link: string;
  title?: string;
  company?: string;
  score?: number;
  details?: string;
  matchedEvidence?: string[];
  missingOrUnclear?: string[];
  coverLetter?: string;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8010";

function collectUrls(value: unknown): string[] {
  const urls = new Set<string>();
  const pattern = /https?:\/\/[^\s"'<>),\]]+/g;

  const visit = (item: unknown) => {
    if (!item) return;
    if (typeof item === "string") {
      for (const match of item.matchAll(pattern)) {
        urls.add(match[0].replace(/[.;]+$/, ""));
      }
      return;
    }
    if (Array.isArray(item)) {
      item.forEach(visit);
      return;
    }
    if (typeof item === "object") {
      Object.values(item as Record<string, unknown>).forEach(visit);
    }
  };

  visit(value);
  return Array.from(urls);
}

function formatToolCall(tool: AgentToolCall) {
  const output = tool.output?.content ?? tool.output;
  if (typeof output === "string") return output;
  return JSON.stringify(output, null, 2);
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function parseJsonFromText(text: string): unknown | null {
  const trimmed = text.trim();
  const fencedMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const unfenced = fencedMatch?.[1]?.trim() ?? trimmed.replace(/^```(?:json)?\s*/i, "").replace(/```$/i, "").trim();
  const candidates = [unfenced, trimmed];
  const objectStart = unfenced.indexOf("{");
  const objectEnd = unfenced.lastIndexOf("}");
  if (objectStart >= 0 && objectEnd > objectStart) {
    candidates.push(unfenced.slice(objectStart, objectEnd + 1));
  }
  const arrayStart = unfenced.indexOf("[");
  const arrayEnd = unfenced.lastIndexOf("]");
  if (arrayStart >= 0 && arrayEnd > arrayStart) {
    candidates.push(unfenced.slice(arrayStart, arrayEnd + 1));
  }

  for (const candidate of candidates) {
    try {
      return JSON.parse(candidate);
    } catch {
      // keep trying candidates
    }
  }
  return null;
}

function numberFrom(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function stringListFrom(value: unknown): string[] | undefined {
  if (!Array.isArray(value)) return undefined;
  return value.filter((item): item is string => typeof item === "string");
}

function jobFromRecord(record: Record<string, unknown>): JobResult | null {
  const linkValue =
    record.job_link ??
    record.url ??
    record.redirect_url ??
    record.link ??
    record.apply_link;
  const link = typeof linkValue === "string" ? linkValue : undefined;
  if (!link) return null;

  return {
    link,
    title: typeof record.title === "string" ? record.title : undefined,
    company: typeof record.company === "string" ? record.company : undefined,
    score: numberFrom(record.match_score ?? record.score),
    details:
      typeof record.extracted_details === "string"
        ? record.extracted_details
        : typeof record.description === "string"
          ? record.description
          : undefined,
    matchedEvidence: stringListFrom(record.matched_cv_evidence),
    missingOrUnclear: stringListFrom(record.missing_or_unclear),
    coverLetter:
      typeof record.cover_letter === "string"
        ? record.cover_letter
        : typeof record.coverLetter === "string"
          ? record.coverLetter
          : undefined,
  };
}

function collectJobResults(data: AgentResponse): JobResult[] {
  const jobsByLink = new Map<string, JobResult>();
  const addJob = (job: JobResult | null) => {
    if (!job) return;
    const current = jobsByLink.get(job.link);
    if (!current || (job.score ?? -1) > (current.score ?? -1)) {
      jobsByLink.set(job.link, job);
    }
  };

  const visit = (value: unknown) => {
    if (!value) return;
    if (typeof value === "string") {
      const parsed = parseJsonFromText(value);
      if (parsed) visit(parsed);
      collectUrls(value).forEach((link) => addJob({ link }));
      return;
    }
    if (Array.isArray(value)) {
      value.forEach(visit);
      return;
    }
    const record = asRecord(value);
    if (!record) return;
    if (Array.isArray(record.jobs)) {
      visit(record.jobs);
      return;
    }
    addJob(jobFromRecord(record));
    Object.values(record).forEach(visit);
  };

  visit(data.result);
  data.tool_calls.forEach((tool) => visit(tool.output?.content ?? tool.output));

  return Array.from(jobsByLink.values()).sort((a, b) => {
    const scoreDiff = (b.score ?? -1) - (a.score ?? -1);
    if (scoreDiff !== 0) return scoreDiff;
    return a.link.localeCompare(b.link);
  });
}

function summarizeAgentResult(result: string, jobs: JobResult[]) {
  const trimmed = result.trim();
  const parsed = trimmed ? parseJsonFromText(trimmed) : null;
  const parsedRecord = asRecord(parsed);
  if (typeof parsedRecord?.summary === "string" && parsedRecord.summary.trim()) {
    return parsedRecord.summary.trim();
  }
  if (Array.isArray(parsed) || parsedRecord?.jobs) {
    if (jobs.length) {
      return `Found ${jobs.length} job link${jobs.length === 1 ? "" : "s"} and sorted them by CV match score.`;
    }
    return "I finished the search, but no usable job links were returned.";
  }

  if (trimmed) {
    if (trimmed.length <= 1200) return trimmed;
    return `${trimmed.slice(0, 1200).trim()}...`;
  }

  if (jobs.length) {
    const scoredCount = jobs.filter((job) => job.score !== undefined).length;
    return scoredCount
      ? `Found ${jobs.length} jobs and sorted them by CV match score. Match reasons are shown in the job list.`
      : `Found ${jobs.length} job link${jobs.length === 1 ? "" : "s"}. I updated the job list on the left.`;
  }

  return "I finished the request, but no job links were returned.";
}

function formatTrace(trace: AgentTrace) {
  const cost =
    trace.estimated_cost_usd > 0
      ? `$${trace.estimated_cost_usd.toFixed(6)}`
      : "$0.000000";

  return [
    `Model: ${trace.model ?? "unknown"}`,
    `Input tokens: ${trace.input_tokens}`,
    `Output tokens: ${trace.output_tokens}`,
    `Total tokens: ${trace.total_tokens}`,
    `Estimated cost: ${cost}`,
    `Tool calls: ${trace.tool_call_count}`,
    `Tool call time: ${trace.tool_call_time_ms} ms`,
    `Total run time: ${trace.total_time_ms} ms`,
  ].join("\n");
}

function safeFilename(value: string) {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 80);
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content:
        "Tell me what role, location, and preferences you want. Upload your CV when you want me to include it in the search context.",
    },
  ]);
  const [input, setInput] = useState("");
  const [cvText, setCvText] = useState("");
  const [cvName, setCvName] = useState("");
  const [cvError, setCvError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [latestJobs, setLatestJobs] = useState<JobResult[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function downloadCoverLetter(job: JobResult) {
    if (!job.coverLetter) return;
    const title = job.title || "job";
    const company = job.company ? ` at ${job.company}` : "";
    const content = `Cover letter for ${title}${company}\n\nJob link: ${job.link}\n\n${job.coverLetter}\n`;
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${safeFilename(`${title}-${job.company || "company"}-cover-letter`) || "cover-letter"}.txt`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  async function handleFileUpload(file: File) {
    setCvName(file.name);
    setCvError("");
    setCvText("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_BASE_URL}/api/cv/extract`, {
        method: "POST",
        body: formData,
      });

      const data = (await response.json()) as CVExtractResponse | { detail?: string };
      if (!response.ok) {
        throw new Error("detail" in data && data.detail ? data.detail : `CV extraction failed with ${response.status}`);
      }

      setCvText((data as CVExtractResponse).text);
    } catch (error) {
      setCvError(error instanceof Error ? error.message : "Could not extract CV text.");
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
    };

    setMessages((current) => [...current, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/api/agent/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input: trimmed,
          cv_text: cvText || null,
        }),
      });

      if (!response.ok) {
        throw new Error(`Backend returned ${response.status}`);
      }

      const data = (await response.json()) as AgentResponse;
      const jobs = collectJobResults(data);
      const links = jobs.map((job) => job.link);
      const summary = summarizeAgentResult(data.result, jobs);

      setLatestJobs(jobs);

      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content: summary,
          links,
          toolCalls: data.tool_calls,
          trace: data.trace ?? null,
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content:
            error instanceof Error
              ? `I could not reach CareerAgent Pro backend: ${error.message}`
              : "I could not reach CareerAgent Pro backend.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <aside className="sidebar" aria-label="CareerAgent Pro controls">
          <div className="brand">
            <div className="brand-mark">
              <BriefcaseBusiness size={22} aria-hidden="true" />
            </div>
            <div>
              <h1>CareerAgent Pro</h1>
              <p>Agentic job intelligence</p>
            </div>
          </div>

          <div className="upload-panel">
            <div className="panel-heading">
              <FileText size={18} aria-hidden="true" />
              <span>CV context</span>
            </div>
            <input
              ref={fileInputRef}
              type="file"
              accept=".txt,.md,.pdf,.doc,.docx"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void handleFileUpload(file);
              }}
            />
            <button type="button" onClick={() => fileInputRef.current?.click()}>
              Upload CV
            </button>
            <p className="file-status">{cvName || "No CV uploaded"}</p>
            {!!cvText && <p className="cv-ready">CV match scoring enabled</p>}
            {!!cvError && <p className="cv-error">{cvError}</p>}
          </div>

          <div className="links-panel">
            <div className="panel-heading">
              <LinkIcon size={18} aria-hidden="true" />
              <span>Captured links</span>
            </div>
            {latestJobs.length ? (
              <ul>
                {latestJobs.map((job, index) => (
                  <li key={job.link}>
                    <span>{index + 1}</span>
                    <div className="job-card">
                      <div className="job-card-top">
                        <div className="job-title-block">
                          <a href={job.link} target="_blank" rel="noreferrer">
                            {job.title || job.link}
                          </a>
                          {job.company && <p>{job.company}</p>}
                        </div>
                        <div className="job-meta-actions">
                          {job.score !== undefined && (
                            <strong>{Math.max(0, Math.min(10, job.score)).toFixed(1)}/10</strong>
                          )}
                          <a className="apply-button" href={job.link} target="_blank" rel="noreferrer">
                            <ExternalLink size={13} aria-hidden="true" />
                            Apply
                          </a>
                        </div>
                      </div>
                      {job.details && <p>{job.details}</p>}
                      {!!job.matchedEvidence?.length && (
                        <div className="reason-block">
                          <b>Matched</b>
                          <ul>
                            {job.matchedEvidence.slice(0, 3).map((reason) => (
                              <li key={reason}>{reason}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {!!job.missingOrUnclear?.length && (
                        <div className="reason-block muted-reasons">
                          <b>Missing or unclear</b>
                          <ul>
                            {job.missingOrUnclear.slice(0, 3).map((reason) => (
                              <li key={reason}>{reason}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {job.coverLetter && (
                        <details className="cover-letter">
                          <summary>Cover letter</summary>
                          <p>{job.coverLetter}</p>
                          <button type="button" onClick={() => downloadCoverLetter(job)}>
                            <Download size={13} aria-hidden="true" />
                            Download
                          </button>
                        </details>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="empty-state">The latest job links from the agent will appear here.</p>
            )}
          </div>
        </aside>

        <section className="chat-panel" aria-label="CareerAgent Pro chat">
          <div className="chat-header">
            <div>
              <h2>Job Search Agent</h2>
              <p>Ask for roles, locations, portals, and filtering preferences.</p>
            </div>
            <span className="status-dot">Backend: {API_BASE_URL}</span>
          </div>

          <div className="messages">
            {messages.map((message) => (
              <article key={message.id} className={`message ${message.role}`}>
                <div className="avatar">
                  {message.role === "assistant" ? <Bot size={18} /> : <UserRound size={18} />}
                </div>
                <div className="bubble">
                  <p>{message.content}</p>
                  {!!message.links?.length && (
                    <div className="job-links">
                      {message.links.map((link) => (
                        <a key={link} href={link} target="_blank" rel="noreferrer">
                          {link}
                        </a>
                      ))}
                    </div>
                  )}
                  {!!message.toolCalls?.length && (
                    <details>
                      <summary>Tool calls</summary>
                      {message.toolCalls.map((tool, index) => (
                        <pre key={`${tool.name}-${index}`}>{tool.name}: {formatToolCall(tool)}</pre>
                      ))}
                    </details>
                  )}
                  {message.trace && (
                    <details>
                      <summary>Trace</summary>
                      <pre>{formatTrace(message.trace)}</pre>
                    </details>
                  )}
                </div>
              </article>
            ))}
            {isLoading && (
              <article className="message assistant">
                <div className="avatar">
                  <Bot size={18} />
                </div>
                <div className="bubble loading">Searching jobs...</div>
              </article>
            )}
          </div>

          <form className="composer" onSubmit={handleSubmit}>
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Search AI engineer jobs in Berlin and return 10 links..."
              rows={3}
            />
            <button type="submit" disabled={isLoading || !input.trim()} aria-label="Send message">
              <Send size={18} aria-hidden="true" />
            </button>
          </form>
        </section>
      </section>
    </main>
  );
}
