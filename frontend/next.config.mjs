/** @type {import('next').NextConfig} */
const nextConfig = {
  // Next writes an AGENTS.md and a CLAUDE.md on first dev start unless told
  // not to. An instruction file that appears on its own and then steers later
  // sessions is not something this repo should acquire as a side effect of
  // starting a dev server.
  agentRules: false,

  // The dashboard is a pure read layer over the API. It holds no simulation
  // state of its own, so there is nothing here to configure beyond the proxy.
  async rewrites() {
    return [{ source: "/api/:path*", destination: "http://127.0.0.1:8000/:path*" }];
  },
};
export default nextConfig;
