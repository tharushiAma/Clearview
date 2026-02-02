# Clearview Website

This directory contains the frontend website for the Clearview sentiment analysis project.

## Tech Stack

- **Framework**: Next.js (React)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui
- **Package Manager**: pnpm

## Quick Start

### Installation

```bash
# Install dependencies
pnpm install
```

### Development

```bash
# Run development server
pnpm dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Build

```bash
# Create production build
pnpm build

# Start production server
pnpm start
```

## Project Structure

```
website/
├── app/          # Next.js app directory (routes)
├── components/   # React components
├── hooks/        # Custom React hooks
├── lib/          # Utility functions
├── public/       # Static assets
├── styles/       # Global styles
└── package.json  # Dependencies
```

## Features

- Interactive sentiment analysis demo
- Multi-aspect visualization
- Real-time XAI explanations
- Responsive design

## Development Notes

- Uses App Router (Next.js 13+)
- Configured with ESLint and Prettier
- TypeScript strict mode enabled

## Deployment

This website can be deployed to Vercel, Netlify, or any Next.js-compatible hosting platform.

```bash
# Production build
pnpm build
```

## Contributing

Please ensure code passes linting and type-checking before committing:

```bash
pnpm lint
pnpm type-check
```
