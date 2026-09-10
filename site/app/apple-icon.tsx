import { ImageResponse } from "next/og";

// Same mark as app/icon.tsx, rendered at Apple's standard touch-icon size with more
// breathing room since it's shown larger (home screen), not squeezed into a 16px tab.
export const size = { width: 180, height: 180 };
export const contentType = "image/png";

const BAR_HEIGHTS = [0.38, 0.62, 1, 0.7, 0.46];
const ACCENT_INDEX = 2;

export default function AppleIcon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "#12151b",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "flex-end",
            justifyContent: "center",
            gap: 12,
            width: 124,
            height: 112,
          }}
        >
          {BAR_HEIGHTS.map((h, i) => (
            <div
              key={i}
              style={{
                width: 16,
                height: Math.round(112 * h),
                borderRadius: 6,
                background: i === ACCENT_INDEX ? "#3fd8c2" : "#caa06a",
              }}
            />
          ))}
        </div>
      </div>
    ),
    { ...size },
  );
}
