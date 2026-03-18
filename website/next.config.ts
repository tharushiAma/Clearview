import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  typescript: {
    ignoreBuildErrors: false,
  },
  async redirects() {
    return [
      {
        source: "/",
        destination: "/demo",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
