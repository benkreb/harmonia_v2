# Harmonia Defense UI

Stage-based, offline React presentation UI for the Harmonia Vision capstone defense.

## Run

```bash
npm install
npm run dev
```

## Static Build

```bash
npm run build
npm run preview
```

The Vite base path is `./`, so `dist/index.html` can also be opened directly when browser security allows local JSON/image access. If `file://` blocks JSON fetches, use `npm run preview` or `npx serve dist`.

## Presenter Controls

- `ArrowRight`, `Space`, `PageDown`: next stage
- `ArrowLeft`, `PageUp`: previous stage
- `1` through `8`: jump to a stage
- `p`: toggle presenter overlay
- `0`: reset zoom

## Data Assets

Assets are generated from `harmonia_v2`:

- `public/data/benchmark.json`
- `public/data/samples_curated.json`
- `public/predictions/sample_*_*.png`
