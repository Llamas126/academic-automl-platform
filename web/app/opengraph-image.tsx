import { ImageResponse } from "next/og";

export const alt = "Academic AutoML Platform";
export const size = {
  width: 1200,
  height: 630,
};
export const contentType = "image/png";

export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          background: "#18181b",
          gap: "24px",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "20px",
          }}
        >
          <div
            style={{
              width: 96,
              height: 96,
              borderRadius: 20,
              background: "#27272a",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            <span
              style={{
                fontSize: 48,
                fontWeight: 700,
                color: "white",
                fontFamily: "sans-serif",
              }}
            >
              ML
            </span>
          </div>
          <span
            style={{
              fontSize: 56,
              fontWeight: 700,
              color: "white",
              fontFamily: "sans-serif",
            }}
          >
            Academic AutoML Platform
          </span>
        </div>
        <span
          style={{
            fontSize: 24,
            color: "#a1a1aa",
            fontFamily: "sans-serif",
          }}
        >
          Automated machine learning with nested cross-validation &amp; SHAP
          explainability
        </span>
      </div>
    ),
    size,
  );
}
