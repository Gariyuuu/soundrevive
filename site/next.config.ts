import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Silences Turbopack's lockfile-detection warning: an unrelated package-lock.json sits at
  // ~/Projects (outside this repo, belongs to some other project) and would otherwise be
  // picked up as a false "workspace root" candidate.
  turbopack: {
    root: path.join(__dirname),
  },
};

export default nextConfig;
