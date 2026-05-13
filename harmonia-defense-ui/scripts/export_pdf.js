import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { PDFDocument } from "pdf-lib";
import puppeteer from "puppeteer";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const distIndex = "file://" + path.resolve(__dirname, "../dist/index.html");
const totalStages = 8;

const browser = await puppeteer.launch({
  headless: "new",
  defaultViewport: { width: 1920, height: 1080 },
});
const page = await browser.newPage();
await page.goto(distIndex, { waitUntil: "networkidle0" });

const pdfPaths = [];
for (let idx = 0; idx < totalStages; idx += 1) {
  await page.evaluate((stageIdx) => window.__goToStage?.(stageIdx), idx);
  await new Promise((resolve) => setTimeout(resolve, 900));
  const pdfPath = path.resolve(__dirname, `../dist/_pdf_stage_${idx}.pdf`);
  await page.pdf({
    path: pdfPath,
    width: "1920px",
    height: "1080px",
    printBackground: true,
    landscape: true,
  });
  pdfPaths.push(pdfPath);
}
await browser.close();

const merged = await PDFDocument.create();
for (const pdfPath of pdfPaths) {
  const src = await PDFDocument.load(await fs.readFile(pdfPath));
  const [pageCopy] = await merged.copyPages(src, [0]);
  merged.addPage(pageCopy);
}

const outPath = path.resolve(__dirname, "../dist/harmonia_defense_backup.pdf");
await fs.writeFile(outPath, await merged.save());
await Promise.all(pdfPaths.map((pdfPath) => fs.rm(pdfPath, { force: true })));
console.log(`Wrote ${outPath}`);
