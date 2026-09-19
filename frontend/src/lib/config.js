// Central configuration -- API endpoints and report letterhead.
// Edit API_BASE_URL when deploying (see README).

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const HOSPITAL = {
  name: 'Meridian Orthopaedic Institute',
  shortName: 'Meridian',
  tagline: 'Precision Diagnostics for Growing Bones',
  department: 'Department of Musculoskeletal Radiology & AI Diagnostics',
  address: 'Academic Research Deployment · GRAZPEDWRI-DX Pediatric Wrist Cohort',
  emblemText: 'M',
};

export const DOCTORS = [
  { name: 'Dr. Ayush Dwivedi', role: 'Project Lead, Model Development', reg: 'Reg. No. 23BCE1539' },
  { name: 'Dr. Kartik Ahlawat', role: 'Co-Investigator, Clinical Validation', reg: 'Reg. No. 23BCE1966' },
];

export const CNN_CLASS_ORDER = ['fracture', 'softtissue_indirect', 'foreign_material', 'bone_lesion'];

export const CNN_CLASS_META = {
  fracture: { label: 'Fracture', color: '#e0475a', accent: 'rose' },
  foreign_material: { label: 'Foreign Material / Metal', color: '#d4a94a', accent: 'gold' },
  softtissue_indirect: { label: 'Soft-Tissue / Indirect Signs', color: '#4a90d9', accent: 'blue' },
  bone_lesion: { label: 'Bone Anomaly / Lesion', color: '#4fd8ac', accent: 'emerald' },
};

// YOLO26 uses the original 9-class GRAZPEDWRI-DX taxonomy.
export const YOLO_CLASS_ORDER = [
  'fracture', 'periostealreaction', 'pronatorsign', 'softtissue',
  'metal', 'boneanomaly', 'bonelesion', 'foreignbody', 'text',
];

export const YOLO_CLASS_META = {
  fracture: { label: 'Fracture', color: '#e0475a' },
  periostealreaction: { label: 'Periosteal Reaction', color: '#4a90d9' },
  pronatorsign: { label: 'Pronator Sign', color: '#7c6fd6' },
  softtissue: { label: 'Soft Tissue Swelling', color: '#3fb8c4' },
  metal: { label: 'Metal', color: '#d4a94a' },
  boneanomaly: { label: 'Bone Anomaly', color: '#4fd8ac' },
  bonelesion: { label: 'Bone Lesion', color: '#e08a3c' },
  foreignbody: { label: 'Foreign Body', color: '#c47ad6' },
  text: { label: 'Text / Marker', color: '#8fa3ad' },
};
