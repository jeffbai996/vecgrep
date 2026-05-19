/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Optional URL to a companion app, surfaced as a link in the header.
   *  Leave unset for OSS builds — no internal hostnames in source. */
  readonly VITE_COMPANION_URL?: string;
  /** Optional label for the companion link. Defaults to "companion". */
  readonly VITE_COMPANION_LABEL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
