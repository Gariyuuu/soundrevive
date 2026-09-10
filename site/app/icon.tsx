import { ImageResponse } from "next/og";

// SoundRevive's mark: a small EQ/waveform silhouette on a dark ground, with one bar picked
// out in a contrasting accent color — the "signal recovered from the noise floor" idea the
// whole project is about, kept simple enough to still read at 16-32px.
export const size = { width: 32, height: 32 };
export const contentType = "image/png";

const BAR_HEIGHTS = [0.38, 0.62, 1, 0.7, 0.46];
const ACCENT_INDEX = 2;

export default function Icon() {
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
          borderRadius: 7,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "flex-end",
            justifyContent: "center",
            gap: 2,
            width: 22,
            height: 20,
          }}
        >
          {BAR_HEIGHTS.map((h, i) => (
            <div
              key={i}
              style={{
                width: 3,
                height: Math.round(20 * h),
                borderRadius: 1,
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
