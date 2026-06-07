const form = document.getElementById("pipeline-form");
const fileInput = document.getElementById("document-file");
const providerSelect = document.getElementById("provider");
const signatureCheckbox = document.getElementById("signature-enabled");
const signatureThreshold = document.getElementById("signature-threshold");
const thresholdValue = document.getElementById("threshold-value");
const fileName = document.getElementById("file-name");
const runButton = document.getElementById("run-button");
const statusChip = document.getElementById("status-chip");
const messages = document.getElementById("messages");
const summaryGrid = document.getElementById("summary-grid");
const businessFindings = document.getElementById("business-findings");
const extractionPreview = document.getElementById("extraction-preview");
const detailsRoot = document.getElementById("details-root");
const resultCard = document.querySelector(".result-card");

const moneyFormatter = new Intl.NumberFormat("en-US", {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const currentDate = new Date();
const initialLedgerYear = currentDate.getFullYear();
const initialLedgerMonth = currentDate.getMonth() + 1;

let supplierLedgerElements = null;
let saveResultElements = null;
let lastCompletedDocumentId = null;

const PAGE_HEADER_PATTERN = /^\[PAGE\s+(\d+)\]\s*$/i;
const DATE_REGEXES = [
  /\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b/g,
  /\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b/g,
];

function setStatusChip(mode, label) {
  statusChip.className = `chip ${mode}`;
  statusChip.textContent = label;
}

function clearMessages() {
  messages.innerHTML = "";
}

function addMessage(type, text) {
  const div = document.createElement("div");
  div.className = `message message-${type}`;
  div.textContent = text;
  messages.appendChild(div);
}

function formatMoney(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return safeString(value) || "-";
  }
  return moneyFormatter.format(numeric);
}

function toPrettyJson(value) {
  return JSON.stringify(value, null, 2);
}

function createSummaryCard(label, value) {
  const card = document.createElement("div");
  card.className = "summary-card";

  const name = document.createElement("div");
  name.className = "summary-label";
  name.textContent = label;

  const content = document.createElement("div");
  content.className = "summary-value";
  content.textContent = value;

  card.appendChild(name);
  card.appendChild(content);
  return card;
}

function createFinding(title, value) {
  const wrap = document.createElement("div");
  wrap.className = "finding";

  const titleEl = document.createElement("div");
  titleEl.className = "finding-title";
  titleEl.textContent = title;

  const valueEl = document.createElement("div");
  valueEl.className = "finding-value";
  valueEl.textContent = value;

  wrap.appendChild(titleEl);
  wrap.appendChild(valueEl);
  return wrap;
}

function createJsonDetails(title, payload, openByDefault = false) {
  const details = document.createElement("details");
  details.open = openByDefault;

  const summary = document.createElement("summary");
  summary.textContent = title;

  const pre = document.createElement("pre");
  pre.textContent = toPrettyJson(payload);

  details.appendChild(summary);
  details.appendChild(pre);
  return details;
}

function createLedgerSection() {
  if (!resultCard || supplierLedgerElements) {
    return supplierLedgerElements;
  }

  const section = document.createElement("details");
  section.className = "ledger-section";
  section.open = false;

  const sectionHead = document.createElement("summary");
  sectionHead.className = "ledger-head";

  const titleWrap = document.createElement("div");

  const title = document.createElement("h3");
  title.textContent = "Supplier Ordered Amounts";

  const subtitle = document.createElement("p");
  subtitle.className = "ledger-subtitle";
  subtitle.textContent =
    "Track monthly ordered amounts from supplier statements already saved in the system.";

  titleWrap.appendChild(title);
  titleWrap.appendChild(subtitle);

  const controls = document.createElement("div");
  controls.className = "ledger-controls";

  const monthSelect = document.createElement("select");
  monthSelect.className = "ledger-control";
  monthSelect.setAttribute("aria-label", "Ledger month");

  const monthNames = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
  ];

  monthNames.forEach((name, index) => {
    const option = document.createElement("option");
    option.value = String(index + 1);
    option.textContent = name;
    if (index + 1 === initialLedgerMonth) {
      option.selected = true;
    }
    monthSelect.appendChild(option);
  });

  const yearInput = document.createElement("input");
  yearInput.type = "number";
  yearInput.className = "ledger-control ledger-year";
  yearInput.setAttribute("aria-label", "Ledger year");
  yearInput.min = "2020";
  yearInput.max = "2100";
  yearInput.step = "1";
  yearInput.value = String(initialLedgerYear);

  const refreshButton = document.createElement("button");
  refreshButton.type = "button";
  refreshButton.className = "ledger-refresh";
  refreshButton.textContent = "Refresh";

  controls.appendChild(monthSelect);
  controls.appendChild(yearInput);
  controls.appendChild(refreshButton);

  sectionHead.appendChild(titleWrap);
  sectionHead.appendChild(controls);

  const body = document.createElement("div");
  body.className = "ledger-body";

  const status = document.createElement("div");
  status.className = "ledger-status";
  status.textContent = "Loading supplier ledger...";

  const list = document.createElement("div");
  list.className = "ledger-list";

  body.appendChild(status);
  body.appendChild(list);

  section.appendChild(sectionHead);
  section.appendChild(body);

  resultCard.insertBefore(section, messages);

  supplierLedgerElements = {
    section,
    body,
    status,
    list,
    monthSelect,
    yearInput,
    refreshButton,
  };

  refreshButton.addEventListener("click", () => {
    refreshSupplierLedger();
  });

  monthSelect.addEventListener("change", () => {
    refreshSupplierLedger();
  });

  yearInput.addEventListener("change", () => {
    refreshSupplierLedger();
  });

  return supplierLedgerElements;
}

function createSaveResultSection() {
  if (!resultCard || saveResultElements) {
    return saveResultElements;
  }

  const section = document.createElement("section");
  section.className = "save-result-section";
  section.hidden = true;

  const title = document.createElement("h3");
  title.textContent = "Save Processed Result";

  const description = document.createElement("p");
  description.className = "save-result-subtitle";
  description.textContent =
    "After reviewing the pipeline result, save this document to the database for analytics and tracking.";

  const actions = document.createElement("div");
  actions.className = "save-result-actions";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "save-result-button";
  button.textContent = "Save To Database";

  const status = document.createElement("div");
  status.className = "save-result-status";

  actions.appendChild(button);
  actions.appendChild(status);

  section.appendChild(title);
  section.appendChild(description);
  section.appendChild(actions);

  resultCard.insertBefore(section, detailsRoot.parentElement);

  saveResultElements = {
    section,
    button,
    status,
  };

  button.addEventListener("click", () => {
    saveLastProcessedResult();
  });

  return saveResultElements;
}

function hideSaveResultSection() {
  const saveSection = createSaveResultSection();
  if (!saveSection) {
    return;
  }

  saveSection.section.hidden = true;
  saveSection.button.disabled = false;
  saveSection.button.textContent = "Save To Database";
  saveSection.status.textContent = "";
  saveSection.status.className = "save-result-status";
}

function showSaveResultSection(message = "This result is not saved to the database yet.") {
  const saveSection = createSaveResultSection();
  if (!saveSection) {
    return;
  }

  saveSection.section.hidden = false;
  saveSection.button.disabled = false;
  saveSection.button.textContent = "Save To Database";
  saveSection.status.textContent = message;
  saveSection.status.className = "save-result-status";
}

function markSaveResultComplete(message) {
  const saveSection = createSaveResultSection();
  if (!saveSection) {
    return;
  }

  saveSection.section.hidden = false;
  saveSection.button.disabled = true;
  saveSection.button.textContent = "Saved";
  saveSection.status.textContent = message;
  saveSection.status.className = "save-result-status save-result-status-ok";
}

function markSaveResultError(message) {
  const saveSection = createSaveResultSection();
  if (!saveSection) {
    return;
  }

  saveSection.section.hidden = false;
  saveSection.button.disabled = false;
  saveSection.button.textContent = "Save To Database";
  saveSection.status.textContent = message;
  saveSection.status.className = "save-result-status save-result-status-error";
}

function createLedgerCard(summary) {
  const card = document.createElement("article");
  card.className = "ledger-card";

  const top = document.createElement("div");
  top.className = "ledger-card-top";

  const supplierName = document.createElement("h4");
  supplierName.className = "ledger-supplier-name";
  supplierName.textContent =
    safeString(summary?.supplier_name) || safeString(summary?.supplier_id) || "Unknown Supplier";

  const supplierId = document.createElement("div");
  supplierId.className = "ledger-supplier-id";
  supplierId.textContent = safeString(summary?.supplier_id) || "-";

  top.appendChild(supplierName);
  top.appendChild(supplierId);

  const ordered = document.createElement("div");
  ordered.className = "ledger-ordered";
  ordered.innerHTML = `
    <span class="ledger-ordered-label">Ordered Amount</span>
    <span class="ledger-ordered-value">${escapeHtml(formatMoney(summary?.ordered_amount))}</span>
  `;

  card.appendChild(top);
  card.appendChild(ordered);
  return card;
}

async function fetchSupplierLedger(year, month) {
  const params = new URLSearchParams({
    year: String(year),
    month: String(month),
  });

  const response = await fetch(
    `/documents/analytics/suppliers/monthly-ledger?${params.toString()}`
  );
  const payload = await parseJsonSafe(response);

  if (!response.ok) {
    throw new Error(
      `Supplier ledger failed (${response.status}): ${payload.detail ?? "unknown error"}`
    );
  }

  return payload;
}

async function saveProcessedResult(documentId) {
  const params = new URLSearchParams({
    document_id: documentId,
  });

  const response = await fetch(`/documents/save-result?${params.toString()}`, {
    method: "POST",
  });
  const payload = await parseJsonSafe(response);

  if (!response.ok) {
    throw new Error(
      `Save failed (${response.status}): ${payload.detail ?? "unknown error"}`
    );
  }

  return payload;
}

async function refreshSupplierLedger() {
  const ledger = createLedgerSection();
  if (!ledger) {
    return;
  }

  const year = Number(ledger.yearInput.value || initialLedgerYear);
  const month = Number(ledger.monthSelect.value || initialLedgerMonth);

  ledger.status.textContent = "Loading supplier ledger...";
  ledger.status.className = "ledger-status";
  ledger.list.innerHTML = "";
  ledger.refreshButton.disabled = true;

  try {
    const payload = await fetchSupplierLedger(year, month);
    const summaries = Array.isArray(payload?.supplier_summaries)
      ? payload.supplier_summaries
      : [];

    if (summaries.length === 0) {
      ledger.status.textContent = `No supplier statement totals found for ${payload?.month_key ?? `${year}-${month}`}.`;
      ledger.status.className = "ledger-status ledger-status-empty";
      return;
    }

    ledger.status.textContent = `Showing supplier totals for ${payload?.month_key ?? `${year}-${month}`}.`;
    ledger.status.className = "ledger-status ledger-status-ok";

    summaries
      .slice()
      .sort((left, right) => Number(right?.ordered_amount || 0) - Number(left?.ordered_amount || 0))
      .forEach((summary) => {
        ledger.list.appendChild(createLedgerCard(summary));
      });
  } catch (error) {
    ledger.status.textContent =
      error instanceof Error ? error.message : "Failed to load supplier ledger.";
    ledger.status.className = "ledger-status ledger-status-error";
  } finally {
    ledger.refreshButton.disabled = false;
  }
}

function safeString(value) {
  if (value === null || value === undefined) {
    return "";
  }
  return String(value).trim();
}

function escapeHtml(value) {
  return safeString(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function extractDatesFromText(text) {
  const value = safeString(text);
  if (!value) {
    return [];
  }

  const matches = [];
  DATE_REGEXES.forEach((regex) => {
    const found = value.match(regex);
    if (found) {
      matches.push(...found);
    }
  });

  return matches;
}

function splitTextByPage(extractedText) {
  const pageMap = {};
  if (!extractedText) {
    return pageMap;
  }

  let currentPage = null;
  const lines = String(extractedText).split(/\r?\n/);

  lines.forEach((line) => {
    const headerMatch = line.match(PAGE_HEADER_PATTERN);
    if (headerMatch) {
      currentPage = Number(headerMatch[1]);
      if (!pageMap[currentPage]) {
        pageMap[currentPage] = [];
      }
      return;
    }

    if (currentPage !== null) {
      pageMap[currentPage].push(line);
    }
  });

  return pageMap;
}

function getDocumentAiExtractedText(processPayload) {
  return safeString(processPayload?.steps?.document_ai?.result?.extracted_text);
}

function getOcrPageTextByPage(processPayload) {
  const rawPages = processPayload?.steps?.document_ai?.result?.raw_output?.primary?.pages;
  const map = {};

  if (!Array.isArray(rawPages)) {
    return map;
  }

  rawPages.forEach((page) => {
    const pageNumber = Number(page?.page_number || 0);
    if (!pageNumber) {
      return;
    }
    map[pageNumber] = safeString(page?.page_text);
  });

  return map;
}

function normalizeArabic(text) {
  return safeString(text)
    .replaceAll("أ", "ا")
    .replaceAll("إ", "ا")
    .replaceAll("آ", "ا")
    .replaceAll("ة", "ه")
    .replaceAll("ى", "ي")
    .replace(/\s+/g, " ")
    .trim();
}

function extractCompanyFromText(text) {
  const lines = safeString(text)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const previousLine = index > 0 ? lines[index - 1] : "";
    const normalized = normalizeArabic(line).toLowerCase();
    const normalizedPrevious = normalizeArabic(previousLine).toLowerCase();

    if (normalized === "pharma" && previousLine) {
      return `${previousLine} PHARMA`;
    }

    if (normalized.includes("فارما") && previousLine) {
      if (normalizedPrevious.includes("الفتح")) {
        return `${previousLine} ${line}`.trim();
      }
    }

    if (normalized.includes("pharma") && normalized.length < 80) {
      return line;
    }

    if (normalized.includes("فارما") && normalized.length < 80) {
      return line;
    }

    if (normalized.includes("الفتح") && normalized.length < 80) {
      return line;
    }
  }

  return null;
}

function getCompanyFromFilePath(processPayload) {
  const sourcePath = safeString(processPayload?.outputs?.source_file_path);
  if (!sourcePath) {
    return null;
  }

  const rawName = sourcePath.split(/[\\/]/).pop() || "";
  const withoutExt = rawName.replace(/\.[^.]+$/, "");
  const cleaned = withoutExt.replace(/^doc_[^_]+_/, "").replaceAll("_", " ").trim();
  return cleaned || null;
}

function getSignatureSummary(processPayload) {
  const signatureStep = processPayload?.steps?.signature;
  if (!signatureStep) {
    return { hasSignature: null, pages: [] };
  }

  if (signatureStep.status === "skipped") {
    return { hasSignature: null, pages: [] };
  }

  const signatureResult = signatureStep.result ?? {};
  const pages = Array.isArray(signatureResult.signature_pages)
    ? signatureResult.signature_pages
    : [];
  const hasSignature =
    pages.length > 0 ||
    signatureResult?.best_page_result?.signature_present === true;

  return { hasSignature, pages };
}

function getSpatialTables(processPayload) {
  const tables = processPayload?.steps?.spatial?.result?.tables;
  return Array.isArray(tables) ? tables : [];
}

function getCandidateInvoicePage(processPayload) {
  const tables = getSpatialTables(processPayload);
  if (tables.length === 0) {
    return null;
  }

  const signaturePages = getSignatureSummary(processPayload).pages;
  if (signaturePages.length > 0) {
    const fromSignature = tables.find(
      (table) =>
        signaturePages.includes(Number(table?.page_number)) &&
        Number(table?.row_count || 0) >= 3
    );
    if (fromSignature) {
      return Number(fromSignature.page_number);
    }
  }

  let best = null;
  tables.forEach((table) => {
    const rowCount = Number(table?.row_count || 0);
    const extractedRowsCount = Array.isArray(table?.extracted_rows)
      ? table.extracted_rows.length
      : 0;
    const score = rowCount - extractedRowsCount;
    if (!best || score > best.score) {
      best = { page: Number(table?.page_number || 0), score };
    }
  });

  return best && best.page ? best.page : null;
}

function getExtractionRouteResult(processPayload) {
  return processPayload?.steps?.extraction?.result ?? null;
}

function getPageExtractions(processPayload) {
  const routeResult = getExtractionRouteResult(processPayload);
  if (!routeResult) {
    return [];
  }

  if (safeString(routeResult.document_type) === "mixed") {
    const pageExtractions = routeResult?.extraction_result?.page_extractions;
    return Array.isArray(pageExtractions) ? pageExtractions : [];
  }

  return [
    {
      page_number: 1,
      page_document_type: safeString(routeResult.document_type) || "unknown",
      extraction_result: routeResult.extraction_result ?? {},
    },
  ];
}

function flattenExtractionRows(processPayload) {
  const routeResult = getExtractionRouteResult(processPayload);
  if (!routeResult) {
    return [];
  }

  const rows = [];
  const topType = safeString(routeResult.document_type) || "unknown";

  if (topType === "mixed") {
    const pageExtractions = routeResult?.extraction_result?.page_extractions ?? [];
    pageExtractions.forEach((pagePayload) => {
      const pageNumber = Number(pagePayload?.page_number || 0);
      const sourceType = safeString(pagePayload?.page_document_type) || "unknown";
      const pageRows = pagePayload?.extraction_result?.rows ?? [];

      pageRows.forEach((row) => {
        rows.push({
          ...row,
          __page_number: pageNumber,
          __source_type: sourceType,
        });
      });
    });
    return rows;
  }

  const directRows = routeResult?.extraction_result?.rows ?? [];
  directRows.forEach((row) => {
    rows.push({
      ...row,
      __page_number: Number(row?.page_number || 0),
      __source_type: topType,
    });
  });

  return rows;
}

function getCompanySummary(processPayload, rows) {
  for (const row of rows) {
    const text = safeString(row.account_name || row.company_name || row.supplier_name);
    if (text) {
      return text;
    }
  }

  const extractedText = getDocumentAiExtractedText(processPayload);
  const fromText = extractCompanyFromText(extractedText);
  if (fromText) {
    return fromText;
  }

  const fromFilePath = getCompanyFromFilePath(processPayload);
  return fromFilePath || "n/a";
}

function getItemCountSummary(processPayload, rows) {
  const directItems = rows.filter((row) => safeString(row.item_name)).length;
  if (directItems > 0) {
    return directItems;
  }

  const candidatePage = getCandidateInvoicePage(processPayload);
  if (!candidatePage) {
    return 0;
  }

  const tables = getSpatialTables(processPayload);
  const table = tables.find((item) => Number(item?.page_number || 0) === candidatePage);
  if (!table) {
    return 0;
  }

  const rowCount = Number(table?.row_count || 0);
  return rowCount > 0 ? rowCount : 0;
}

function getDateSummary(processPayload, rows) {
  const pageTextMap = splitTextByPage(getDocumentAiExtractedText(processPayload));
  const ocrPageTextMap = getOcrPageTextByPage(processPayload);
  const candidatePage = getCandidateInvoicePage(processPayload);

  if (candidatePage) {
    let candidateText = "";
    if (pageTextMap[candidatePage]) {
      candidateText = pageTextMap[candidatePage].join("\n");
    } else if (ocrPageTextMap[candidatePage]) {
      candidateText = ocrPageTextMap[candidatePage];
    }

    const candidateDates = [...new Set(extractDatesFromText(candidateText))];
    if (candidateDates.length > 0) {
      return {
        date: candidateDates[0],
        count: candidateDates.length,
      };
    }
  }

  const directDates = [];

  rows.forEach((row) => {
    if (safeString(row.movement_date)) {
      directDates.push(safeString(row.movement_date));
    }
    if (safeString(row.date)) {
      directDates.push(safeString(row.date));
    }
    if (safeString(row.description)) {
      directDates.push(...extractDatesFromText(row.description));
    }
  });

  if (directDates.length > 0) {
    const uniqueDates = [...new Set(directDates)];
    return { date: uniqueDates[0], count: uniqueDates.length };
  }

  let pageText = "";
  if (candidatePage && pageTextMap[candidatePage]) {
    pageText = pageTextMap[candidatePage].join("\n");
  } else if (candidatePage && ocrPageTextMap[candidatePage]) {
    pageText = ocrPageTextMap[candidatePage];
  } else {
    pageText = getDocumentAiExtractedText(processPayload);
  }

  const pageDates = extractDatesFromText(pageText);
  const uniqueDates = [...new Set(pageDates)];

  return {
    date: uniqueDates[0] || "n/a",
    count: uniqueDates.length,
  };
}

function renderSummaries(uploadPayload, processPayload) {
  summaryGrid.innerHTML = "";

  const rows = flattenExtractionRows(processPayload);
  const signature = getSignatureSummary(processPayload);
  const company = getCompanySummary(processPayload, rows);
  const itemCount = getItemCountSummary(processPayload, rows);
  const dateSummary = getDateSummary(processPayload, rows);

  const cards = [
    createSummaryCard("Company", company),
    createSummaryCard("Date", dateSummary.date),
    createSummaryCard("Items", String(itemCount)),
    createSummaryCard(
      "Sign",
      signature.hasSignature === null ? "Not run" : signature.hasSignature ? "Yes" : "No"
    ),
    createSummaryCard("Document Type", processPayload?.steps?.classification?.result?.document_type ?? "n/a"),
    createSummaryCard("Document ID", uploadPayload?.document_id ?? "n/a"),
    createSummaryCard("Date Count", String(dateSummary.count)),
    createSummaryCard("Signature Pages", signature.pages.length ? signature.pages.join(", ") : "none"),
  ];

  cards.forEach((card) => summaryGrid.appendChild(card));
}

function renderBusinessFindings(processPayload) {
  businessFindings.innerHTML = "";

  const status = safeString(processPayload?.status);
  if (!status) {
    return;
  }

  if (status === "pipeline_failed") {
    businessFindings.appendChild(
      createFinding("Failed Step", safeString(processPayload?.failed_step) || "unknown")
    );
    businessFindings.appendChild(
      createFinding("Failure Reason", safeString(processPayload?.error) || "No error message.")
    );
    return;
  }

  const validation = safeString(processPayload?.outputs?.validation_status);
  if (validation) {
    businessFindings.appendChild(createFinding("Validation", validation));
  }
}

function parseStatementDescription(descriptionText) {
  const parts = safeString(descriptionText)
    .split("|")
    .map((part) => part.trim())
    .filter(Boolean);

  return {
    date: parts[0] || "-",
    ref: parts[1] || "-",
    note: parts[2] || "-",
  };
}

function rowToPreviewItem(row) {
  const sourceType = safeString(row.__source_type) || safeString(row.document_type);
  if (sourceType === "monthly_statement") {
    const parsed = parseStatementDescription(row.description);
    return {
      page: safeString(row.__page_number) || safeString(row.page_number) || "-",
      type: sourceType || "-",
      date: safeString(row.date) || parsed.date,
      subject: safeString(row.reference_number) || parsed.ref,
      item: safeString(row.note) || parsed.note,
      qty: safeString(row.debit) || safeString(row.credit) || "-",
      total: safeString(row.balance) || "-",
    };
  }

  const dateFromFields = safeString(row.movement_date) || safeString(row.date);
  if (dateFromFields) {
    return {
      page: safeString(row.__page_number) || safeString(row.page_number) || "-",
      type: sourceType || "-",
      date: dateFromFields,
      subject: safeString(row.account_name) || "-",
      item: safeString(row.item_name) || "-",
      qty: safeString(row.quantity) || safeString(row.debit) || safeString(row.credit) || "-",
      total: safeString(row.total) || safeString(row.balance) || safeString(row.price) || "-",
    };
  }

  const parsed = parseStatementDescription(row.description);
  return {
    page: safeString(row.__page_number) || safeString(row.page_number) || "-",
    type: sourceType || "-",
    date: parsed.date,
    subject: parsed.ref,
    item: parsed.note,
    qty: safeString(row.debit) || safeString(row.credit) || "-",
    total: safeString(row.balance) || "-",
  };
}

function getPageText(processPayload, pageNumber) {
  const pageTextMap = splitTextByPage(getDocumentAiExtractedText(processPayload));
  if (pageTextMap[pageNumber]) {
    return pageTextMap[pageNumber].join("\n");
  }

  const ocrPageTextMap = getOcrPageTextByPage(processPayload);
  return safeString(ocrPageTextMap[pageNumber]);
}

function extractNumberTokens(text) {
  const matches = safeString(text).match(/\b\d[\d,]*(?:\.\d+)?\b/g);
  return matches ? matches : [];
}

function extractMonthlyFooterSummary(processPayload, pageNumber) {
  const pageText = getPageText(processPayload, pageNumber);
  if (!pageText) {
    return null;
  }

  const lines = pageText
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const footerTokens = [];
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const line = lines[index];
    if (/\d{4}[/-]\d{1,2}[/-]\d{1,2}/.test(line)) {
      continue;
    }

    const tokens = extractNumberTokens(line);
    tokens.forEach((token) => footerTokens.push(token));
    if (footerTokens.length >= 3) {
      break;
    }
  }

  if (footerTokens.length < 3) {
    return null;
  }

  const orderedTokens = footerTokens.slice(0, 3).reverse();

  return {
    total_credit: orderedTokens[0] || "-",
    total_debit: orderedTokens[1] || "-",
    current_balance: orderedTokens[2] || "-",
  };
}

function appendSectionTitle(container, text) {
  const title = document.createElement("h3");
  title.textContent = text;
  container.appendChild(title);
}

function appendRowsTable(container, headers, rowValues) {
  const wrapper = document.createElement("div");
  wrapper.className = "rows-table-wrap";

  const table = document.createElement("table");
  table.className = "rows-table";

  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  headers.forEach((header) => {
    const th = document.createElement("th");
    th.textContent = header;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);

  const tbody = document.createElement("tbody");
  rowValues.forEach((values) => {
    const tr = document.createElement("tr");
    values.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = safeString(value) || "-";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  table.appendChild(thead);
  table.appendChild(tbody);
  wrapper.appendChild(table);
  container.appendChild(wrapper);
}

function renderMonthlyStatementSection(processPayload, pagePayload) {
  const pageNumber = Number(pagePayload?.page_number || 0);
  const rows = Array.isArray(pagePayload?.extraction_result?.rows)
    ? pagePayload.extraction_result.rows
    : [];

  appendSectionTitle(
    extractionPreview,
    `Page ${pageNumber} · Monthly Statement`
  );

  appendRowsTable(
    extractionPreview,
    ["Date", "Reference", "Note", "Debit", "Credit", "Balance"],
    rows.slice(0, 20).map((row) => [
      row.date,
      row.reference_number,
      row.note,
      row.debit,
      row.credit,
      row.balance,
    ])
  );

  const footerSummary = extractMonthlyFooterSummary(processPayload, pageNumber);
  if (!footerSummary) {
    return;
  }

  const summaryTitle = document.createElement("h4");
  summaryTitle.textContent = "Closing Summary";
  extractionPreview.appendChild(summaryTitle);

  appendRowsTable(
    extractionPreview,
    ["الإجمالي دائن له", "الإجمالي مدين عليه", "الرصيد الحالي"],
    [[
      footerSummary.total_credit,
      footerSummary.total_debit,
      footerSummary.current_balance,
    ]]
  );
}

function renderDailyInvoiceSection(pagePayload) {
  const pageNumber = Number(pagePayload?.page_number || 0);
  const rows = Array.isArray(pagePayload?.extraction_result?.rows)
    ? pagePayload.extraction_result.rows
    : [];

  appendSectionTitle(
    extractionPreview,
    `Page ${pageNumber} · Daily Invoice`
  );

  appendRowsTable(
    extractionPreview,
    ["Item", "Quantity", "Price", "Discount", "Total"],
    rows.slice(0, 20).map((row) => [
      row.item_name,
      row.quantity,
      row.price,
      row.discount,
      row.total,
    ])
  );
}

function renderGenericPreview(processPayload) {
  const title = document.createElement("h3");
  title.textContent = "Preview (simplified)";
  extractionPreview.appendChild(title);

  const rows = flattenExtractionRows(processPayload);
  appendRowsTable(
    extractionPreview,
    ["Page", "Type", "Date", "Ref/Account", "Item/Note", "Qty/Debit", "Total/Balance"],
    rows.slice(0, 12).map((row) => {
      const p = rowToPreviewItem(row);
      return [
        p.page,
        p.type,
        p.date,
        p.subject,
        p.item,
        p.qty,
        p.total,
      ];
    })
  );
}

function renderExtractionPreview(processPayload) {
  extractionPreview.innerHTML = "";

  const pageExtractions = getPageExtractions(processPayload);
  if (pageExtractions.length === 0) {
    const empty = document.createElement("div");
    empty.className = "rows-empty";
    empty.textContent = "No extracted rows available.";
    extractionPreview.appendChild(empty);
    return;
  }

  const hasSpecializedPages = pageExtractions.some((pagePayload) => {
    const pageType = safeString(pagePayload?.page_document_type);
    return pageType === "monthly_statement" || pageType === "daily_invoice";
  });

  if (!hasSpecializedPages) {
    renderGenericPreview(processPayload);
    return;
  }

  pageExtractions.forEach((pagePayload) => {
    const pageType = safeString(pagePayload?.page_document_type);
    if (pageType === "monthly_statement") {
      renderMonthlyStatementSection(processPayload, pagePayload);
      return;
    }

    if (pageType === "daily_invoice") {
      renderDailyInvoiceSection(pagePayload);
      return;
    }

    renderGenericPreview(processPayload);
  });
}

function renderAdvancedDetails(uploadPayload, processPayload) {
  detailsRoot.innerHTML = "";
  detailsRoot.appendChild(createJsonDetails("Upload Response", uploadPayload, false));
  detailsRoot.appendChild(createJsonDetails("Pipeline Response", processPayload, false));
}

async function parseJsonSafe(response) {
  const text = await response.text();
  if (!text) {
    return {};
  }
  try {
    return JSON.parse(text);
  } catch {
    return { raw_response: text };
  }
}

async function uploadDocument(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/documents/upload", {
    method: "POST",
    body: formData,
  });
  const payload = await parseJsonSafe(response);

  if (!response.ok) {
    throw new Error(
      `Upload failed (${response.status}): ${payload.detail ?? "unknown error"}`
    );
  }

  return payload;
}

async function processDocument(documentId) {
  const params = new URLSearchParams({
    document_id: documentId,
    document_ai_provider: providerSelect.value,
    run_signature_detection: signatureCheckbox.checked ? "true" : "false",
    save_to_database: "false",
  });

  if (signatureCheckbox.checked) {
    params.set("signature_conf_threshold", signatureThreshold.value);
  }

  const response = await fetch(`/documents/process?${params.toString()}`, {
    method: "POST",
  });
  const payload = await parseJsonSafe(response);

  if (!response.ok) {
    throw new Error(
      `Process failed (${response.status}): ${payload.detail ?? "unknown error"}`
    );
  }

  return payload;
}

async function saveLastProcessedResult() {
  if (!lastCompletedDocumentId) {
    markSaveResultError("No processed document is ready to save.");
    return;
  }

  const saveSection = createSaveResultSection();
  saveSection.button.disabled = true;
  saveSection.button.textContent = "Saving...";
  saveSection.status.textContent = "Saving processed result to database...";
  saveSection.status.className = "save-result-status";

  try {
    const payload = await saveProcessedResult(lastCompletedDocumentId);
    markSaveResultComplete(
      `Saved to database. supplier_id=${safeString(payload?.supplier_id) || "n/a"}`
    );
    addMessage("info", "Processed result saved to database.");
    refreshSupplierLedger();
  } catch (error) {
    markSaveResultError(
      error instanceof Error ? error.message : "Failed to save processed result."
    );
    addMessage(
      "error",
      error instanceof Error ? error.message : "Failed to save processed result."
    );
  }
}

async function runPipeline(event) {
  event.preventDefault();
  clearMessages();
  summaryGrid.innerHTML = "";
  businessFindings.innerHTML = "";
  extractionPreview.innerHTML = "";
  detailsRoot.innerHTML = "";
  hideSaveResultSection();
  lastCompletedDocumentId = null;

  const selectedFile = fileInput.files && fileInput.files[0];
  if (!selectedFile) {
    addMessage("error", "Please select a file first.");
    return;
  }

  runButton.disabled = true;
  setStatusChip("chip-running", "running");

  try {
    addMessage("info", "Uploading document...");
    const uploadPayload = await uploadDocument(selectedFile);

    const documentId = uploadPayload?.document_id;
    if (!documentId) {
      throw new Error("Upload response is missing document_id.");
    }

    addMessage("info", `Upload completed. document_id=${documentId}`);
    addMessage("info", "Running OCR pipeline...");

    const processPayload = await processDocument(documentId);
    lastCompletedDocumentId = documentId;

    renderSummaries(uploadPayload, processPayload);
    renderBusinessFindings(processPayload);
    renderExtractionPreview(processPayload);
    renderAdvancedDetails(uploadPayload, processPayload);

    const isSuccess = String(processPayload?.status || "").startsWith("pipeline_completed");
    if (isSuccess) {
      setStatusChip("chip-ok", "completed");
      addMessage("info", "Pipeline completed.");
      showSaveResultSection("Review the result, then save it to the database if you want to track it.");
    } else {
      setStatusChip("chip-error", "needs review");
      addMessage("error", `Pipeline ended with status: ${processPayload?.status ?? "unknown"}`);
      showSaveResultSection("This result is still available to save after review.");
    }
  } catch (error) {
    setStatusChip("chip-error", "failed");
    addMessage("error", error instanceof Error ? error.message : "Unexpected error.");
    hideSaveResultSection();
  } finally {
    runButton.disabled = false;
  }
}

fileInput.addEventListener("change", () => {
  const selectedFile = fileInput.files && fileInput.files[0];
  fileName.textContent = selectedFile
    ? `${selectedFile.name} (${Math.round(selectedFile.size / 1024)} KB)`
    : "No file selected";
});

signatureThreshold.addEventListener("input", () => {
  thresholdValue.textContent = Number(signatureThreshold.value).toFixed(2);
});

form.addEventListener("submit", runPipeline);

createLedgerSection();
createSaveResultSection();
hideSaveResultSection();
refreshSupplierLedger();
