// Tables to .tsf: reads a CSV / text table or an Excel workbook (.xlsx) and writes the Monash .tsf text the engine reads.
// Works in a page, in a worker and under Node (no DOM is used).
//
//   TsfConvert.isTsf(name, buffer)                  -> true when the file is already a .tsf file
//   TsfConvert.convert(name, buffer, { frequency }) -> Promise<{ name, text, report }>; throws TsfConvert.ConvertError
//
// Accepted layouts (found from the header names, else from the contents):
//   long     one row per observation: a series column, a date column and a value column
//   wide     one row per date: a date column, then one column per series
//   columns  no dates at all: every numeric column is a series (the frequency has to be chosen)
// Dates are read as calendar dates without time zones. A series must be regular at the dataset's frequency; dates that
// are absent inside a series become missing values, as do empty cells.
(function (root) {
  "use strict";

  class ConvertError extends Error {}

  const FREQUENCIES = ["4_seconds", "minutely", "10_minutes", "15_minutes", "half_hourly", "hourly", "daily", "weekly",
                       "monthly", "quarterly", "yearly"];
  const STEP_SECONDS = { "4_seconds": 4, minutely: 60, "10_minutes": 600, "15_minutes": 900, half_hourly: 1800, hourly: 3600,
                         daily: 86400, weekly: 604800 };
  const STEP_MONTHS = { monthly: 1, quarterly: 3, yearly: 12 };
  const MISSING = new Set(["", "?", "na", "n/a", "nan", "null", "none", "-", "#n/a"]);
  const NAMES = {
    series: ["series", "series_name", "seriesname", "id", "unique_id", "item", "item_id", "name", "sku", "product", "store",
             "key", "seri", "seri_adi", "urun"],
    date: ["date", "ds", "time", "timestamp", "datetime", "period", "month", "week", "year", "day", "tarih", "donem", "ay",
           "yil", "hafta", "gun"],
    value: ["value", "y", "target", "sales", "demand", "quantity", "qty", "amount", "volume", "deger", "satis", "talep",
            "miktar"],
  };
  const MAX_LISTED = 5;
  const DAY = 86400000;

  // ------------------------------------------------------------------------------------------------ text
  const TR = { "ı": "i", "İ": "I", "ş": "s", "Ş": "S", "ğ": "g", "Ğ": "G", "ü": "u", "Ü": "U", "ö": "o", "Ö": "O", "ç": "c", "Ç": "C" };
  function ascii(s) {
    return String(s).replace(/[ıİşŞğĞüÜöÖçÇ]/g, (c) => TR[c]).normalize("NFKD").replace(/[\u0300-\u036f]/g, "");
  }
  function sanitizeName(s) {
    return ascii(s).trim().replace(/\s+/g, "_").replace(/[^A-Za-z0-9_.\-]/g, "_");
  }
  function headerKey(s) {
    return ascii(s == null ? "" : s).trim().toLowerCase().replace(/[\s\-.]+/g, "_");
  }

  function decodeText(buffer) {
    const bytes = new Uint8Array(buffer);
    if (bytes.length >= 2 && ((bytes[0] === 0xff && bytes[1] === 0xfe) || (bytes[0] === 0xfe && bytes[1] === 0xff)))
      return new TextDecoder(bytes[0] === 0xff ? "utf-16le" : "utf-16be").decode(bytes);
    try {
      return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    } catch (e) {
      return new TextDecoder("windows-1254").decode(bytes);   // text saved by a spreadsheet in the local code page
    }
  }

  // ------------------------------------------------------------------------------------------------ CSV
  function detectDelimiter(text) {
    const lines = [];
    let pos = 0;
    while (lines.length < 30 && pos < text.length) {
      let end = text.indexOf("\n", pos);
      if (end < 0) end = text.length;
      const line = text.slice(pos, end).replace(/\r$/, "");
      if (line.trim()) lines.push(line);
      pos = end + 1;
    }
    let best = ",", bestScore = 0;
    for (const d of ["\t", ";", ",", "|"]) {
      let min = Infinity, same = true, first = -1;
      for (const line of lines) {
        let n = 0, quoted = false;
        for (let i = 0; i < line.length; i++) {
          const c = line[i];
          if (c === '"') quoted = !quoted;
          else if (c === d && !quoted) n++;
        }
        if (first < 0) first = n; else if (n !== first) same = false;
        if (n < min) min = n;
      }
      if (!lines.length || min === 0 || min === Infinity) continue;
      const score = min * (same ? 4 : 1);          // the same count on every line is strong evidence
      if (score > bestScore) { bestScore = score; best = d; }
    }
    return best;
  }

  function parseCsv(text, delimiter) {
    if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
    const rows = [];
    let row = [], cell = "", quoted = false, wasQuoted = false;
    const n = text.length;
    for (let i = 0; i < n; i++) {
      const c = text[i];
      if (quoted) {
        if (c === '"') {
          if (text[i + 1] === '"') { cell += '"'; i++; } else quoted = false;
        } else cell += c;
      } else if (c === '"' && cell === "") { quoted = true; wasQuoted = true; }
      else if (c === delimiter) { row.push(cell); cell = ""; wasQuoted = false; }
      else if (c === "\n" || c === "\r") {
        if (c === "\r" && text[i + 1] === "\n") i++;
        row.push(cell); cell = ""; wasQuoted = false;
        rows.push(row); row = [];
      } else cell += c;
    }
    if (cell !== "" || wasQuoted || row.length) { row.push(cell); rows.push(row); }
    return rows;
  }

  // ------------------------------------------------------------------------------------------------ Excel (.xlsx)
  function u16(b, o) { return b[o] | (b[o + 1] << 8); }
  function u32(b, o) { return (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)) >>> 0; }

  function zipEntries(bytes) {
    let e = -1;
    for (let i = bytes.length - 22; i >= 0 && i >= bytes.length - 22 - 65536; i--)
      if (u32(bytes, i) === 0x06054b50) { e = i; break; }
    if (e < 0) throw new ConvertError("The Excel file could not be opened (it is not a .xlsx workbook).");
    const count = u16(bytes, e + 10);
    let p = u32(bytes, e + 16);
    const out = new Map(), dec = new TextDecoder("utf-8");
    for (let k = 0; k < count && p + 46 <= bytes.length && u32(bytes, p) === 0x02014b50; k++) {
      const method = u16(bytes, p + 10), csize = u32(bytes, p + 20), nlen = u16(bytes, p + 28), xlen = u16(bytes, p + 30),
            clen = u16(bytes, p + 32), local = u32(bytes, p + 42);
      const name = dec.decode(bytes.subarray(p + 46, p + 46 + nlen));
      out.set(name, { method, csize, local });
      p += 46 + nlen + xlen + clen;
    }
    return out;
  }

  async function zipText(bytes, entries, name) {
    const en = entries.get(name) || entries.get(name.replace(/^\//, ""));
    if (!en) return null;
    const start = en.local + 30 + u16(bytes, en.local + 26) + u16(bytes, en.local + 28);
    const raw = bytes.subarray(start, start + en.csize);
    if (en.method === 0) return new TextDecoder("utf-8").decode(raw);
    if (en.method !== 8) throw new ConvertError("The Excel file uses a compression this page cannot read; save it as CSV.");
    if (typeof DecompressionStream === "undefined")
      throw new ConvertError("This browser cannot open Excel files; save the sheet as CSV and choose that file.");
    const stream = new Blob([raw]).stream().pipeThrough(new DecompressionStream("deflate-raw"));
    return new TextDecoder("utf-8").decode(await new Response(stream).arrayBuffer());
  }

  const ENT = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'" };
  function unxml(s) {
    if (s.indexOf("&") < 0) return s;
    return s.replace(/&(#x[0-9a-fA-F]+|#\d+|\w+);/g, (m, k) =>
      k[0] === "#" ? String.fromCodePoint(k[1] === "x" ? parseInt(k.slice(2), 16) : parseInt(k.slice(1), 10)) : (ENT[k] || m));
  }
  function attr(s, name) {
    const m = new RegExp("(?:^|\\s)" + name + '\\s*=\\s*(?:"([^"]*)"|\'([^\']*)\')').exec(s);
    return m ? unxml(m[1] !== undefined ? m[1] : m[2]) : null;
  }
  function texts(xml) {      // the <t> runs of a shared or inline string, without the phonetic ones
    let out = "";
    const re = /<t\b[^>]*?(?:\/>|>([\s\S]*?)<\/t>)/g;
    let m;
    xml = xml.replace(/<rPh\b[\s\S]*?<\/rPh>/g, "");
    while ((m = re.exec(xml))) out += unxml(m[1] || "");
    return out;
  }
  function isDateFormat(code) {
    const c = code.replace(/"[^"]*"/g, "").replace(/\[[^\]]*\]/g, "").replace(/\\./g, "");
    return /[dmyhs]/i.test(c) && !/general/i.test(c);
  }
  const BUILTIN_DATE = new Set([14, 15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 45, 46, 47, 50,
                                51, 52, 53, 54, 55, 56, 57, 58]);

  function colIndex(ref) {
    let c = 0, i = 0;
    for (; i < ref.length; i++) {
      const k = ref.charCodeAt(i);
      if (k >= 65 && k <= 90) c = c * 26 + (k - 64); else if (k >= 97 && k <= 122) c = c * 26 + (k - 96); else break;
    }
    return [c - 1, parseInt(ref.slice(i), 10) - 1];
  }

  function sheetRows(xml, shared, dateStyle, date1904) {
    const rows = [];
    const rowRe = /<row\b([^>]*?)(?:\/>|>([\s\S]*?)<\/row>)/g, cellRe = /<c\b([^>]*?)(?:\/>|>([\s\S]*?)<\/c>)/g;
    let rm, next = 0;
    while ((rm = rowRe.exec(xml))) {
      const rAttr = attr(rm[1], "r");
      const ri = rAttr ? parseInt(rAttr, 10) - 1 : next;
      next = ri + 1;
      if (!rm[2]) continue;
      const row = [];
      let cm, cnext = 0;
      cellRe.lastIndex = 0;
      while ((cm = cellRe.exec(rm[2]))) {
        const a = cm[1], body = cm[2] || "", ref = attr(a, "r");
        const ci = ref ? colIndex(ref)[0] : cnext;
        cnext = ci + 1;
        if (ci < 0 || ci > 16383) continue;
        const t = attr(a, "t");
        const vm = /<v\b[^>]*>([\s\S]*?)<\/v>/.exec(body);
        const v = vm ? unxml(vm[1]) : null;
        let val = null;
        if (t === "s") val = v == null ? null : (shared[parseInt(v, 10)] ?? null);
        else if (t === "inlineStr") val = texts(body);
        else if (t === "str") val = v;
        else if (t === "b" || t === "e") val = null;
        else if (t === "d") val = v;
        else if (v != null && v !== "") {
          const x = Number(v);
          if (Number.isFinite(x)) {
            const s = attr(a, "s");
            if (s != null && dateStyle[parseInt(s, 10)])
              // serial numbers below 60 sit before Excel's non-existent 29 February 1900
              val = { ms: Math.round((x - 25569 + (date1904 ? 1462 : x < 60 ? 1 : 0)) * 86400) * 1000 };
            else val = x;
          }
        }
        if (typeof val === "string" && val.trim() === "") val = null;
        if (val != null) { while (row.length < ci) row.push(null); row[ci] = val; }
      }
      if (row.length) { while (rows.length < ri) rows.push([]); rows[ri] = row; }
    }
    return rows;
  }

  async function readWorkbook(buffer) {
    const bytes = new Uint8Array(buffer);
    const entries = zipEntries(bytes);
    const wb = await zipText(bytes, entries, "xl/workbook.xml");
    if (!wb) throw new ConvertError("The Excel file could not be opened (no workbook inside). Old .xls files are not read: " +
                                    "save the file as .xlsx or CSV.");
    const rels = (await zipText(bytes, entries, "xl/_rels/workbook.xml.rels")) || "";
    const target = {};
    for (const m of rels.matchAll(/<Relationship\b([^>]*?)\/?>/g)) target[attr(m[1], "Id")] = attr(m[1], "Target");
    const sst = await zipText(bytes, entries, "xl/sharedStrings.xml");
    const shared = [];
    if (sst) for (const m of sst.matchAll(/<si\s*\/>|<si\b[^>]*>([\s\S]*?)<\/si>/g)) shared.push(m[1] ? texts(m[1]) : "");
    const styles = (await zipText(bytes, entries, "xl/styles.xml")) || "";
    const custom = {};
    for (const m of styles.matchAll(/<numFmt\b([^>]*?)\/?>/g)) custom[attr(m[1], "numFmtId")] = attr(m[1], "formatCode") || "";
    const dateStyle = [];
    const xfs = /<cellXfs\b[^>]*>([\s\S]*?)<\/cellXfs>/.exec(styles);
    if (xfs) for (const m of xfs[1].matchAll(/<xf\b([^>]*?)(?:\/>|>)/g)) {
      const id = attr(m[1], "numFmtId") || "0";
      dateStyle.push(id in custom ? isDateFormat(custom[id]) : BUILTIN_DATE.has(parseInt(id, 10)));
    }
    const date1904 = /<workbookPr\b[^>]*date1904\s*=\s*["'](1|true)["']/.test(wb);
    const sheets = [];
    for (const m of wb.matchAll(/<sheet\b([^>]*?)\/?>/g)) {
      const name = attr(m[1], "name"), t = target[attr(m[1], "r:id")];
      if (!t) continue;
      const path = t[0] === "/" ? t.slice(1) : "xl/" + t.replace(/^\.\//, "");
      sheets.push({ name, load: async () => sheetRows((await zipText(bytes, entries, path)) || "", shared, dateStyle, date1904) });
    }
    return sheets;
  }

  // ------------------------------------------------------------------------------------------------ numbers and dates
  function parseNumber(v, decimalComma) {
    if (v == null) return NaN;
    if (typeof v === "number") return v;
    if (typeof v !== "string") return undefined;
    let s = v.replace(/[\s\u00a0]/g, "");
    if (MISSING.has(s.toLowerCase())) return NaN;
    if (!/^[-+]?[\d.,]+(?:[eE][-+]?\d+)?$/.test(s) || !/\d/.test(s)) return undefined;
    const dot = s.lastIndexOf("."), comma = s.lastIndexOf(",");
    if (comma >= 0 && dot >= 0) s = comma > dot ? s.replace(/\./g, "").replace(",", ".") : s.replace(/,/g, "");
    else if (comma >= 0) {
      if (s.indexOf(",") !== comma) s = s.replace(/,/g, "");                  // 1,234,567
      else if (decimalComma || !/^[-+]?\d{1,3},\d{3}$/.test(s)) s = s.replace(",", ".");
      else s = s.replace(",", "");                                           // 1,234 in a comma-separated file
    } else if (dot >= 0 && s.indexOf(".") !== dot) s = s.replace(/\./g, "");    // 1.234.567
    const x = Number(s);
    return Number.isFinite(x) ? x : undefined;
  }

  const MONTHS = {};
  [["jan january ocak oca", 1], ["feb february subat şubat sub şub", 2], ["mar march mart", 3], ["apr april nisan nis", 4],
   ["may mayis mayıs", 5], ["jun june haziran haz", 6], ["jul july temmuz tem", 7], ["aug august agustos ağustos agu ağu", 8],
   ["sep sept september eylul eylül eyl", 9], ["oct october ekim eki", 10], ["nov november kasim kasım kas", 11],
   ["dec december aralik aralık ara", 12]].forEach(([w, m]) => w.split(" ").forEach((k) => (MONTHS[k] = m)));

  const R_YMD = /^(\d{4})[-\/.](\d{1,2})[-\/.](\d{1,2})(?:[ T]+(\d{1,2})[:\-](\d{2})(?:[:\-](\d{2})(?:[.,]\d+)?)?)?\s*(?:Z|[+-]\d{2}:?\d{2})?$/;
  const R_XYY = /^(\d{1,2})[-\/.](\d{1,2})[-\/.](\d{4})(?:[ T,]+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AaPp][Mm])?)?$/;
  const R_YM = /^(\d{4})[-\/.](\d{1,2})$/, R_MY = /^(\d{1,2})[-\/.](\d{4})$/, R_Y = /^(\d{4})$/, R_YMD8 = /^(\d{4})(\d{2})(\d{2})$/;
  const R_YQ = /^(\d{4})[-\s\/.]?[QqÇç]([1-4])$/, R_QY = /^[QqÇç]([1-4])[-\s\/.]?(\d{4})$/;
  const R_MON_Y = /^(\p{L}+)[-\s.,\/]+(\d{4})$/u, R_Y_MON = /^(\d{4})[-\s.,\/]+(\p{L}+)$/u;
  const R_D_MON_Y = /^(\d{1,2})[-\s.,\/]+(\p{L}+)[-\s.,\/]+(\d{4})$/u, R_MON_D_Y = /^(\p{L}+)\s+(\d{1,2}),?\s+(\d{4})$/u;

  function utc(y, mo, d, h, mi, s) {
    if (y < 1000 || y > 2999) return null;
    const t = Date.UTC(y, mo - 1, d, h || 0, mi || 0, s || 0);
    const k = new Date(t);
    return k.getUTCFullYear() === y && k.getUTCMonth() === mo - 1 && k.getUTCDate() === d && (h || 0) < 24 && (mi || 0) < 60 ? t : null;
  }

  // -> milliseconds (UTC) or null. `monthFirst` settles 03/04/2020. `plain` is set for a bare number (2020, 20200131).
  function parseDate(v, monthFirst, flags) {
    if (v == null) return null;
    if (typeof v === "object") return typeof v.ms === "number" ? v.ms : null;
    if (typeof v === "number") {
      if (Number.isInteger(v) && v >= 1000 && v <= 2999) { if (flags) flags.plain = true; return Date.UTC(v, 0, 1); }
      return null;
    }
    const s = String(v).trim();
    let m;
    if ((m = R_YMD.exec(s))) return utc(+m[1], +m[2], +m[3], +(m[4] || 0), +(m[5] || 0), +(m[6] || 0));
    if ((m = R_XYY.exec(s))) {
      let h = +(m[4] || 0);
      if (m[7]) { const pm = /p/i.test(m[7]); h = (h % 12) + (pm ? 12 : 0); }
      const a = +m[1], b = +m[2];
      if (flags) { if (a > 12) flags.firstOver = true; if (b > 12) flags.secondOver = true; (flags.first ||= new Set()).add(a); (flags.second ||= new Set()).add(b); }
      return monthFirst ? utc(+m[3], a, b, h, +(m[5] || 0), +(m[6] || 0)) : utc(+m[3], b, a, h, +(m[5] || 0), +(m[6] || 0));
    }
    if ((m = R_YM.exec(s))) return utc(+m[1], +m[2], 1);
    if ((m = R_MY.exec(s))) return utc(+m[2], +m[1], 1);
    if ((m = R_YQ.exec(s))) return utc(+m[1], 3 * (+m[2] - 1) + 1, 1);
    if ((m = R_QY.exec(s))) return utc(+m[2], 3 * (+m[1] - 1) + 1, 1);
    if ((m = R_Y.exec(s))) { if (flags) flags.plain = true; return utc(+m[1], 1, 1); }
    if ((m = R_YMD8.exec(s))) { if (flags) flags.plain = true; return utc(+m[1], +m[2], +m[3]); }
    const mon = (w) => MONTHS[w.toLowerCase()] || MONTHS[ascii(w).toLowerCase()];
    if ((m = R_MON_Y.exec(s)) && mon(m[1])) return utc(+m[2], mon(m[1]), 1);
    if ((m = R_Y_MON.exec(s)) && mon(m[2])) return utc(+m[1], mon(m[2]), 1);
    if ((m = R_D_MON_Y.exec(s)) && mon(m[2])) return utc(+m[3], mon(m[2]), +m[1]);
    if ((m = R_MON_D_Y.exec(s)) && mon(m[1])) return utc(+m[3], mon(m[1]), +m[2]);
    return null;
  }

  // A whole column as dates. -> { ms: [...], ok, filled, plain, monthFirst, ambiguous }
  function dateColumn(cells) {
    const flags = {};
    for (const c of cells) if (typeof c === "string") parseDate(c, false, flags);
    let monthFirst = false, ambiguous = false;
    if (flags.first) {
      if (flags.firstOver) monthFirst = false;
      else if (flags.secondOver) monthFirst = true;
      else { monthFirst = flags.second.size === 1 && flags.first.size > 1; ambiguous = flags.first.size > 1 && flags.second.size > 1; }
    }
    const ms = new Array(cells.length);
    let ok = 0, filled = 0, plain = 0;
    for (let i = 0; i < cells.length; i++) {
      const c = cells[i];
      if (c == null || (typeof c === "string" && c.trim() === "")) { ms[i] = null; continue; }
      filled++;
      const f = {};
      const t = parseDate(c, monthFirst, f);
      ms[i] = t;
      if (t != null) { ok++; if (f.plain) plain++; }
    }
    return { ms, ok, filled, plain, monthFirst, ambiguous, slashed: !!flags.first };
  }

  // ------------------------------------------------------------------------------------------------ the table
  function cleanRows(rows) {
    const out = [];
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i] || [];
      let any = false;
      for (const c of r) if (c != null && !(typeof c === "string" && c.trim() === "")) { any = true; break; }
      if (any) out.push({ cells: r, line: i + 1 });
    }
    return out;
  }
  const blank = (c) => c == null || (typeof c === "string" && c.trim() === "");
  function colLetter(j) {
    let s = "";
    for (j = j + 1; j > 0; j = Math.floor((j - 1) / 26)) s = String.fromCharCode(65 + ((j - 1) % 26)) + s;
    return s;
  }

  function interpret(rows, opt) {
    const R = cleanRows(rows);
    if (R.length < 2) throw new ConvertError("The file holds no table: at least a few rows of numbers are needed.");
    let ncol = 0;
    for (const r of R) ncol = Math.max(ncol, r.cells.length);
    const first = R[0].cells;
    let hasHeader = true, anyHead = false;
    for (let j = 0; j < ncol; j++) {
      const c = first[j];
      if (blank(c)) continue;
      anyHead = true;
      if (typeof c !== "string" || parseNumber(c, opt.decimalComma) !== undefined || parseDate(c, false) != null) { hasHeader = false; break; }
    }
    hasHeader = hasHeader && anyHead;
    const head = hasHeader ? first : [];
    const body = hasHeader ? R.slice(1) : R;
    if (!body.length) throw new ConvertError("The file holds only a header row.");
    const where = (i, j) => (opt.spreadsheet ? "cell " + colLetter(j) + body[i].line : "line " + body[i].line + ", column " + (j + 1));

    const cols = [];
    for (let j = 0; j < ncol; j++) {
      const cells = body.map((r) => (j < r.cells.length ? r.cells[j] : null));
      let filled = 0, num = 0;
      for (const c of cells) {
        if (blank(c)) continue;
        filled++;
        const x = parseNumber(c, opt.decimalComma);
        if (x !== undefined) num++;
      }
      const key = headerKey(head[j]);
      cols.push({ j, cells, filled, num, key, title: blank(head[j]) ? "" : String(head[j]).trim(), dates: null });
    }
    const role = (kind) => cols.find((c) => c.filled && NAMES[kind].includes(c.key));
    const asDates = (c) => c.dates || (c.dates = dateColumn(c.cells));
    const increasing = (ms) => { let p = -Infinity; for (const t of ms) { if (t == null) continue; if (t <= p) return false; p = t; } return true; };
    const isDateCol = (c, named) => {
      if (!c.filled) return false;
      const d = asDates(c);
      if (d.ok < Math.max(1, Math.ceil(0.9 * d.filled))) return false;
      if (d.plain === d.ok) return named || (c.j === 0 && increasing(d.ms));   // bare numbers: years only where it is plain
      return true;
    };

    let dateCol = role("date");
    if (dateCol && !isDateCol(dateCol, true)) {
      const d = asDates(dateCol);
      const i = d.ms.findIndex((t, k) => t == null && !blank(dateCol.cells[k]));
      throw new ConvertError("The column '" + dateCol.title + "' should hold dates, but " + where(i, dateCol.j) + " ('" +
        String(dateCol.cells[i]).slice(0, 30) + "') could not be read as one. Use dates such as 2024-01-31, 31.01.2024 or 2024-01.");
    }
    if (!dateCol) dateCol = cols.find((c) => isDateCol(c, false)) || null;
    let seriesCol = role("series");
    if (seriesCol === dateCol) seriesCol = null;
    let valueCol = role("value");
    if (valueCol === dateCol || valueCol === seriesCol) valueCol = null;

    const repeated = (() => {
      if (!dateCol) return false;
      const seen = new Set();
      for (const t of asDates(dateCol).ms) { if (t == null) continue; if (seen.has(t)) return true; seen.add(t); }
      return false;
    })();
    if (!seriesCol && dateCol && repeated) {         // long layout without a telling header: the text column names the series
      const texty = cols.filter((c) => c !== dateCol && c.filled && c.num < c.filled);
      if (texty.length >= 1) seriesCol = texty[0];
      else throw new ConvertError("The same date appears on several rows, so the file seems to hold several series, but no " +
        "column names the series. Add a column called 'series', or use one column per series.");
    }
    // a column of numbers; the odd cell that is not one is reported with its place instead of dropping the column
    const numeric = (c) => c.filled > 0 && c.num >= Math.max(1, Math.ceil(0.6 * c.filled));
    const ignored = [];
    const out = { layout: "", series: [], ignored, dateInfo: dateCol ? asDates(dateCol) : null };
    const errors = [];
    let nErr = 0;
    const bad = (msg) => { if (nErr++ < MAX_LISTED) errors.push(msg); };

    if (seriesCol && dateCol) {
      out.layout = "long";
      if (!valueCol) valueCol = cols.find((c) => c !== dateCol && c !== seriesCol && numeric(c));
      if (!valueCol) throw new ConvertError("No column of numbers was found beside the series and date columns.");
      for (const c of cols) if (c !== dateCol && c !== seriesCol && c !== valueCol && c.filled) ignored.push(c.title || "column " + (c.j + 1));
      const d = asDates(dateCol), groups = new Map();
      for (let i = 0; i < body.length; i++) {
        const nm = seriesCol.cells[i];
        if (blank(nm) && blank(dateCol.cells[i]) && blank(valueCol.cells[i])) continue;
        if (blank(nm)) { bad(where(i, seriesCol.j) + ": the series name is empty"); continue; }
        if (d.ms[i] == null) { bad(where(i, dateCol.j) + ": " + (blank(dateCol.cells[i]) ? "the date is empty" : "'" + String(dateCol.cells[i]).slice(0, 30) + "' is not a date")); continue; }
        const x = parseNumber(valueCol.cells[i], opt.decimalComma);
        if (x === undefined) { bad(where(i, valueCol.j) + ": '" + String(valueCol.cells[i]).slice(0, 30) + "' is not a number"); continue; }
        const key = String(nm).trim();
        let g = groups.get(key);
        if (!g) groups.set(key, (g = { name: key, t: [], v: [] }));
        g.t.push(d.ms[i]);
        g.v.push(x);
      }
      out.series = [...groups.values()];
    } else if (dateCol) {
      out.layout = "wide";
      const d = asDates(dateCol);
      const vcols = cols.filter((c) => c !== dateCol && numeric(c));
      for (const c of cols) if (c !== dateCol && c.filled && !vcols.includes(c)) ignored.push(c.title || "column " + (c.j + 1));
      if (!vcols.length) throw new ConvertError("No column of numbers was found beside the date column.");
      const keep = [];
      for (let i = 0; i < body.length; i++) {
        if (d.ms[i] == null) {
          if (blank(dateCol.cells[i])) { if (vcols.some((c) => !blank(c.cells[i]))) bad(where(i, dateCol.j) + ": the date is empty"); }
          else bad(where(i, dateCol.j) + ": '" + String(dateCol.cells[i]).slice(0, 30) + "' is not a date");
          continue;
        }
        keep.push(i);
      }
      for (const c of vcols) {
        const s = { name: c.title || (vcols.length === 1 ? "series" : "series_" + (out.series.length + 1)), t: [], v: [] };
        for (const i of keep) {
          const x = parseNumber(c.cells[i], opt.decimalComma);
          if (x === undefined) { bad(where(i, c.j) + ": '" + String(c.cells[i]).slice(0, 30) + "' is not a number"); continue; }
          s.t.push(d.ms[i]);
          s.v.push(x);
        }
        out.series.push(s);
      }
    } else {
      out.layout = "columns";
      const vcols = cols.filter(numeric);
      for (const c of cols) if (c.filled && !vcols.includes(c)) ignored.push(c.title || "column " + (c.j + 1));
      if (!vcols.length) throw new ConvertError("No dates and no columns of numbers were found. The page reads tables with one row " +
        "per date (a date column, then one column per series) or one row per observation (series, date, value).");
      for (const c of vcols) {
        const s = { name: c.title || (vcols.length === 1 ? "series" : "series_" + (out.series.length + 1)), t: null, v: [] };
        for (let i = 0; i < body.length; i++) {
          const x = parseNumber(c.cells[i], opt.decimalComma);
          if (x === undefined) { bad(where(i, c.j) + ": '" + String(c.cells[i]).slice(0, 30) + "' is not a number"); continue; }
          s.v.push(x);
        }
        out.series.push(s);
      }
    }
    if (nErr) throw new ConvertError("The table could not be read: " + errors.join("; ") +
      (nErr > MAX_LISTED ? "; and " + (nErr - MAX_LISTED) + " more" : "") + ".");
    return out;
  }

  // ------------------------------------------------------------------------------------------------ frequency, regular grid
  function describeGap(sec) {
    if (sec < 120) return Math.round(sec) + " seconds";
    if (sec < 7200) return Math.round(sec / 60) + " minutes";
    if (sec < 2 * 86400) return +(sec / 3600).toFixed(1) + " hours";
    return +(sec / 86400).toFixed(1) + " days";
  }

  function inferFrequency(series) {
    const gaps = [];
    for (const s of series) {
      if (gaps.length > 5000) break;
      const t = [...new Set(s.t)].sort((a, b) => a - b);
      for (let i = 1; i < t.length && i < 400; i++) gaps.push((t[i] - t[i - 1]) / 1000);
    }
    if (!gaps.length) return { frequency: null, reason: "every series has a single observation" };
    // the step that occurs most often among the recognised ones (absent dates make longer gaps, which do not count);
    // FREQUENCIES runs from fine to coarse, so a tie goes to the finer step
    const count = {};
    for (const g of gaps) {
      const d = g / 86400;
      let f = null;
      for (const k of Object.keys(STEP_SECONDS)) if (Math.abs(g - STEP_SECONDS[k]) <= STEP_SECONDS[k] * 0.02) f = k;
      if (!f) f = d >= 28 && d <= 31 ? "monthly" : d >= 89 && d <= 92 ? "quarterly" : d >= 365 && d <= 366 ? "yearly" : null;
      if (f) count[f] = (count[f] || 0) + 1;
    }
    let best = null;
    for (const f of FREQUENCIES) if (count[f] && (!best || count[f] > count[best])) best = f;
    if (best) return { frequency: best };
    gaps.sort((a, b) => a - b);
    return { frequency: null, reason: "consecutive dates are usually " + describeGap(gaps[gaps.length >> 1]) + " apart" };
  }

  const pad = (n, w) => String(n).padStart(w || 2, "0");
  function stamp(ms) {
    const d = new Date(ms);
    return d.getUTCFullYear() + "-" + pad(d.getUTCMonth() + 1) + "-" + pad(d.getUTCDate()) + " " + pad(d.getUTCHours()) + "-" +
      pad(d.getUTCMinutes()) + "-" + pad(d.getUTCSeconds());
  }
  function shortDate(ms, frequency) {
    const d = new Date(ms), y = d.getUTCFullYear(), mo = pad(d.getUTCMonth() + 1);
    if (frequency === "yearly") return String(y);
    if (frequency === "quarterly") return y + " Q" + (Math.floor(d.getUTCMonth() / 3) + 1);
    if (frequency === "monthly") return y + "-" + mo;
    const day = y + "-" + mo + "-" + pad(d.getUTCDate());
    return STEP_SECONDS[frequency] < 86400 ? day + " " + pad(d.getUTCHours()) + ":" + pad(d.getUTCMinutes()) : day;
  }

  // One series on the regular grid of the frequency. -> { start (ms), values, filled } ; throws ConvertError
  function regular(s, frequency) {
    const n = s.t.length;
    const idx = new Array(n);
    let startMs;
    if (frequency in STEP_MONTHS) {
      const step = STEP_MONTHS[frequency];
      let min = Infinity;
      for (let i = 0; i < n; i++) {
        const d = new Date(s.t[i]);
        idx[i] = Math.floor((d.getUTCFullYear() * 12 + d.getUTCMonth()) / step);
        if (idx[i] < min) min = idx[i];
      }
      for (let i = 0; i < n; i++) idx[i] -= min;
      startMs = Date.UTC(Math.floor((min * step) / 12), (min * step) % 12, 1);
    } else {
      const step = STEP_SECONDS[frequency] * 1000;
      const t = step >= DAY ? s.t.map((x) => Math.floor(x / DAY) * DAY) : s.t;
      let min = Infinity;
      for (const x of t) if (x < min) min = x;
      for (let i = 0; i < n; i++) {
        const k = (t[i] - min) / step;
        if (Math.abs(k - Math.round(k)) > 1e-6)
          throw new ConvertError("Series '" + s.name + "': its dates do not follow a " + frequency.replace("_", " ") + " rhythm (" +
            shortDate(t[i], frequency) + " does not fit, counting from " + shortDate(min, frequency) + "). Choose another frequency if this one is wrong.");
        idx[i] = Math.round(k);
      }
      startMs = min;
    }
    let len = 0;
    for (const k of idx) if (k + 1 > len) len = k + 1;
    if (len > 5e6 || len > 200 * n + 1000)
      throw new ConvertError("Series '" + s.name + "': its dates span " + len.toLocaleString("en") + " " + frequency.replace("_", " ") +
        " steps for " + n + " observations. A date is probably wrong, or the frequency is.");
    const values = new Float64Array(len).fill(NaN), seen = new Uint8Array(len);
    for (let i = 0; i < n; i++) {
      if (seen[idx[i]]) throw new ConvertError("Series '" + s.name + "': " + shortDate(s.t[i], frequency) +
        " appears more than once. Each series needs one value per period; sum or average the rows first" +
        (frequency in STEP_MONTHS ? "" : ", or choose a finer frequency") + ".");
      seen[idx[i]] = 1;
      values[idx[i]] = s.v[i];
    }
    return { startMs, values, filled: len - n };
  }

  function trim(values) {      // drop leading and trailing missing values
    let a = 0, b = values.length;
    while (a < b && Number.isNaN(values[a])) a++;
    while (b > a && Number.isNaN(values[b - 1])) b--;
    return { a, b };
  }

  function advance(ms, k, frequency) {
    if (!k) return ms;
    if (frequency in STEP_MONTHS) {
      const d = new Date(ms), m = d.getUTCFullYear() * 12 + d.getUTCMonth() + k * STEP_MONTHS[frequency];
      return Date.UTC(Math.floor(m / 12), m % 12, 1);
    }
    return ms + k * STEP_SECONDS[frequency] * 1000;
  }

  function build(table, name, opt) {
    const report = { layout: table.layout, ignored: table.ignored, sheet: opt.sheet || null, delimiter: opt.delimiter || null,
                     notes: [] };
    let frequency = opt.frequency || opt.metaFrequency || null;
    report.frequencySource = opt.frequency ? "chosen" : opt.metaFrequency ? "file" : "detected";
    if (frequency && !FREQUENCIES.includes(frequency)) throw new ConvertError("Unknown frequency '" + frequency + "'.");
    if (table.layout !== "columns" && !frequency) {
      const f = inferFrequency(table.series);
      if (!f.frequency) {
        const e = new ConvertError("The frequency of the data could not be worked out (" + f.reason + "). Choose it below.");
        e.needsFrequency = true;
        throw e;
      }
      frequency = f.frequency;
    }
    if (table.layout === "columns") {
      report.noDates = true;
      if (!frequency) { frequency = "monthly"; report.frequencySource = "assumed"; }
    }
    report.frequency = frequency;

    const used = new Map(), out = [];
    let filled = 0, missing = 0, renamed = 0, dropped = 0, first = Infinity, last = -Infinity, observations = 0;
    for (const s of table.series) {
      let startMs, values;
      if (table.layout === "columns") {
        startMs = Date.UTC(2000, 0, 1);
        values = Float64Array.from(s.v);
      } else {
        if (!s.t.length) { dropped++; continue; }
        const r = regular(s, frequency);
        startMs = r.startMs; values = r.values;
        filled += r.filled;
      }
      const { a, b } = trim(values);
      if (b <= a) { dropped++; continue; }
      if (a > 0 || b < values.length) { values = values.subarray(a, b); startMs = advance(startMs, a, frequency); }
      let miss = 0;
      for (const x of values) if (Number.isNaN(x)) miss++;
      missing += miss;
      let nm = sanitizeName(s.name);
      if (!nm || /^_+$/.test(nm)) nm = "series_" + (out.length + 1);
      if (nm !== String(s.name)) renamed++;
      const k = used.get(nm) || 0;
      used.set(nm, k + 1);
      if (k) { nm = nm + "_" + (k + 1); renamed++; }
      observations += values.length;
      first = Math.min(first, startMs);
      last = Math.max(last, advance(startMs, values.length - 1, frequency));
      out.push({ name: nm, startMs, values });
    }
    if (!out.length) throw new ConvertError("No series with at least one number was found in the file.");
    const lengths = new Set(out.map((s) => s.values.length));
    Object.assign(report, { nSeries: out.length, observations, missing, filled: Math.max(0, filled), renamed, dropped,
      lenMin: Math.min(...lengths), lenMax: Math.max(...lengths), first: shortDate(first, frequency), last: shortDate(last, frequency),
      horizon: opt.metaHorizon || null });
    if (table.dateInfo && table.dateInfo.slashed)
      report.dateOrder = (table.dateInfo.monthFirst ? "month/day/year" : "day/month/year") + (table.dateInfo.ambiguous ? " (assumed)" : "");

    const relation = sanitizeName(opt.relation || name.replace(/\.[^.]+$/, "")) || "user_dataset";
    const lines = ["# Dataset Information", "# Converted from " + ascii(name).replace(/[^\x20-\x7e]/g, "_") + " by Forecast Studio (browser edition).",
      "#", "@relation " + relation, "@attribute series_name string", "@attribute start_timestamp date", "@frequency " + frequency];
    if (report.horizon) lines.push("@horizon " + report.horizon);
    lines.push("@missing " + (missing > 0), "@equallength " + (lengths.size === 1), "@data");
    for (const s of out) {
      const parts = new Array(s.values.length);
      for (let i = 0; i < parts.length; i++) parts[i] = Number.isNaN(s.values[i]) ? "?" : String(s.values[i]);
      lines.push(s.name + ":" + stamp(s.startMs) + ":" + parts.join(","));
    }
    return { name: relation + ".tsf", text: lines.join("\n") + "\n", report };
  }

  // ------------------------------------------------------------------------------------------------ entry points
  function isTsf(name, buffer) {
    if (/\.tsf$/i.test(name)) return true;
    if (/\.(csv|tsv|xlsx|xlsm|xls)$/i.test(name)) return false;
    const head = new TextDecoder("utf-8").decode(new Uint8Array(buffer, 0, Math.min(buffer.byteLength, 262144)));
    return /^@data\s*$/im.test(head) && /^@relation\b/im.test(head);
  }

  async function convert(name, buffer, options) {
    const opt = Object.assign({}, options || {});
    const bytes = new Uint8Array(buffer, 0, Math.min(8, buffer.byteLength));
    const zip = bytes[0] === 0x50 && bytes[1] === 0x4b;
    if (!zip && bytes[0] === 0xd0 && bytes[1] === 0xcf)
      throw new ConvertError("Old Excel files (.xls) are not read. Save the file as .xlsx or CSV and choose that file.");
    let rows;
    if (zip) {
      const sheets = await readWorkbook(buffer);
      if (!sheets.length) throw new ConvertError("The workbook has no sheets.");
      opt.spreadsheet = true;
      const byName = (n) => sheets.find((s) => s.name === n);
      const meta = byName("Meta"), longS = byName("Veri_Uzun"), wideS = byName("Veri_Genis");
      if (meta && (longS || wideS)) {                       // the workbook template of the desktop application
        const kv = {};
        for (const r of await meta.load()) if (r && !blank(r[0])) kv[String(r[0]).trim().toLowerCase()] = r[1];
        if (!blank(kv.frequency)) {
          const f = String(kv.frequency).trim().toLowerCase();
          if (!FREQUENCIES.includes(f)) throw new ConvertError("Meta sheet: frequency '" + f + "' is not valid. Valid values: " + FREQUENCIES.join(", ") + ".");
          opt.metaFrequency = f;
        }
        const h = Number(kv.horizon);
        if (Number.isInteger(h) && h > 0) opt.metaHorizon = h;
        if (!blank(kv.dataset_name)) opt.relation = String(kv.dataset_name);
        const lr = longS ? await longS.load() : [], wr = wideS ? await wideS.load() : [];
        const nl = cleanRows(lr).length, nw = cleanRows(wr).length;
        if (nl > 1 && nw > 1) throw new ConvertError("Both Veri_Uzun and Veri_Genis hold data. Use only one of them.");
        if (nl <= 1 && nw <= 1) throw new ConvertError("Veri_Uzun and Veri_Genis are both empty. Enter the data in one of them.");
        rows = nl > 1 ? lr : wr;
        opt.sheet = nl > 1 ? "Veri_Uzun" : "Veri_Genis";
      } else {
        for (const s of sheets) {
          const r = await s.load();
          if (cleanRows(r).length >= 2) { rows = r; opt.sheet = s.name; break; }
        }
        if (!rows) throw new ConvertError("No sheet of the workbook holds a table.");
      }
    } else {
      const text = decodeText(buffer);
      if (text.indexOf("\u0000") >= 0) throw new ConvertError("The file is not a text table. The page reads .tsf, CSV and Excel (.xlsx) files.");
      const d = detectDelimiter(text);
      opt.delimiter = d;
      opt.decimalComma = d !== ",";
      rows = parseCsv(text, d);
    }
    return build(interpret(rows, opt), name, opt);
  }

  const api = { FREQUENCIES, ConvertError, isTsf, convert, parseDate, parseNumber, advance, shortDate };
  root.TsfConvert = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : globalThis);
