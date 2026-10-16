// One colour per hospital, reused by every chart, row and legend so a node is recognisable at a glance.
export const HOSPITALS = {
  1: { name: "Mercy General", scanner: "3T, low noise", color: "#0B7A75" },
  2: { name: "St. Aurelia", scanner: "1.5T, noisier", color: "#B8661A" },
  3: { name: "Nordic Neuro", scanner: "3T, large lesions", color: "#4B4FB3" },
};

export const hospitalMeta = (id) =>
  HOSPITALS[id] ?? { name: `Hospital ${id}`, scanner: "", color: "#5C6B75" };
