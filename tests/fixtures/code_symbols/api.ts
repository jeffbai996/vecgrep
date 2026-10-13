import { Injectable } from "./di";
import type { Request, Response } from "./http";
import * as log from "./log";

export interface Handler {
  handle(req: Request): Promise<Response>;
}

export enum Mode {
  Fast,
  Safe,
}

export type Route = { path: string; handler: Handler };

/** Serves orders. */
@Injectable()
export class OrderService implements Handler {
  private cache = new Map<string, Response>();

  async handle(req: Request): Promise<Response> {
    const key = req.url;
    function remember(res: Response): Response {
      return res;
    }
    return remember(await this.run(key));
  }

  run(key: string): Promise<Response> {
    log.info(key);
    return Promise.resolve({ status: 200 } as Response);
  }

  onError = (err: Error): void => {
    log.error(err.message);
  };
}

export class AuditService {
  run(key: string): void {
    log.info(`audit ${key}`);
  }
}

export const healthCheck = async (req: Request): Promise<Response> => {
  return { status: 204 } as Response;
};

const internalHelper = function (value: number): number {
  return value * 2;
};

export default function createRouter(routes: Route[]): Map<string, Handler> {
  return new Map(routes.map((r) => [r.path, r.handler]));
}
