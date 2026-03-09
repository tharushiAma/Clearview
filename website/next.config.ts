import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Ensure we strictly fail on errors
  typescript: {
    ignoreBuildErrors: false,
  },
};

export default nextConfig;
