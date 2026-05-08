import 'reflect-metadata'
import { NestFactory } from '@nestjs/core'
import type { VercelRequest, VercelResponse } from '@vercel/node'
import { AppModule } from '../src/app.module'

let appPromise: Promise<any> | null = null

async function getApp(): Promise<any> {
  if (appPromise) return appPromise

  appPromise = NestFactory.create(AppModule, {
    logger: ['error', 'warn'],
  }).then(async (app) => {
    await app.init()
    return app
  })

  return appPromise
}

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    const app = await getApp()
    const httpAdapter = app.getHttpAdapter()
    const instance = httpAdapter.getInstance()

    instance(req, res)
  } catch (err) {
    console.error('Handler error:', err)
    res.status(500).json({ error: 'Internal server error' })
  }
}
