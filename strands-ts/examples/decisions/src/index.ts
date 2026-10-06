import { BedrockModel, LLMDecisionModel, Uncertain } from '@strands-agents/sdk'

async function main() {
  const decision = new LLMDecisionModel(new BedrockModel())

  const result = await decision.ask('Hello', {
    color: {
      instructions: 'What color is the fruit?',
      choices: ['red', 'green', 'blue'] as const,
    },
    isFruit: {
      instructions: 'Is the thing talked about a fruit?',
      choices: 'boolean',
      uncertainOptions: { allow: false },
    },
  })

  // result.answers.color   is 'red' | 'green' | 'blue' | Uncertain
  // result.answers.isFruit is boolean
  console.log('color:  ', result.answers.color)
  console.log('isFruit:', result.answers.isFruit)
  console.log('usage:  ', result.usage)
  console.log('latency:', `${result.metadata.latencyMs}ms`)
}

await main().catch(console.error)
