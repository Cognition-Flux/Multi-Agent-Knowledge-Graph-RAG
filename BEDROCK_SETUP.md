# AWS Bedrock Setup Guide

## Configuración de Permisos IAM

Tu usuario IAM necesita permisos para invocar modelos en AWS Bedrock.

### Opción 1: Usar AWS CLI
```bash
# 1. Crear la política
aws iam create-policy \
    --policy-name BedrockInvokePolicy \
    --policy-document file://bedrock-policy.json

# 2. Adjuntar la política a tu usuario
aws iam attach-user-policy \
    --user-name sea-crawler-archivos \
    --policy-arn arn:aws:iam::443073691211:policy/BedrockInvokePolicy
```

### Opción 2: Usar la Consola de AWS

1. Ve a **IAM** → **Policies** → **Create Policy**
2. Selecciona **JSON** y pega el contenido de `bedrock-policy.json`
3. Nombra la política como `BedrockInvokePolicy`
4. Ve a **IAM** → **Users** → **sea-crawler-archivos**
5. En **Permissions** → **Add permissions** → **Attach existing policies**
6. Busca y selecciona `BedrockInvokePolicy`

### Opción 3: Política Simplificada (más permisos)

Si prefieres una política más simple con acceso completo a Bedrock:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "bedrock:*",
            "Resource": "*"
        }
    ]
}
```

## Verificar Regiones con Bedrock

Asegúrate de que tu región tiene Bedrock disponible:

```bash
# Listar modelos disponibles en tu región
aws bedrock list-foundation-models --region us-east-1
```

Regiones con Bedrock disponible:
- `us-east-1` (N. Virginia)
- `us-west-2` (Oregon)
- `eu-west-1` (Ireland)
- `ap-northeast-1` (Tokyo)
- `ap-southeast-1` (Singapore)

## Modelos Disponibles en Bedrock

### Claude (Anthropic)
- `anthropic.claude-3-5-sonnet-20240620-v1:0` - Más capaz
- `anthropic.claude-3-haiku-20240307-v1:0` - Más rápido
- `anthropic.claude-3-opus-20240229-v1:0` - Más potente

### Llama (Meta)
- `meta.llama3-1-70b-instruct-v1:0`
- `meta.llama3-1-8b-instruct-v1:0`

### Titan (Amazon)
- `amazon.titan-text-express-v1`
- `amazon.titan-text-lite-v1`

## Uso en el Código

```python
from src.utils import get_llm

# Usar Claude 3.5 Sonnet (por defecto)
llm = get_llm(provider="bedrock")

# Usar Claude 3 Haiku (más rápido y económico)
llm = get_llm(provider="bedrock", model="anthropic.claude-3-haiku-20240307-v1:0")

# Usar Llama 3.1 70B
llm = get_llm(provider="bedrock", model="meta.llama3-1-70b-instruct-v1:0")

# Invocar el modelo
response = llm.invoke("¿Cuál es la capital de Francia?")
print(response.content)
```

## Variables de Entorno Necesarias

```bash
# En tu archivo .env
AWS_ACCESS_KEY_ID=tu_access_key
AWS_SECRET_ACCESS_KEY=tu_secret_key
AWS_REGION=us-east-1  # o tu región preferida
```

## Troubleshooting

### Error: AccessDeniedException
- Verifica que tu usuario tiene la política IAM correcta
- Confirma que el modelo está disponible en tu región

### Error: ResourceNotFoundException
- El modelo especificado no está disponible en tu región
- Verifica el nombre exacto del modelo

### Error: ValidationException
- Revisa que los parámetros sean válidos
- Algunos modelos tienen límites específicos de tokens

## Costos

Ten en cuenta los costos de Bedrock:
- Claude 3.5 Sonnet: ~$3 por millón de tokens de entrada
- Claude 3 Haiku: ~$0.25 por millón de tokens de entrada
- Consulta la [página de precios de AWS Bedrock](https://aws.amazon.com/bedrock/pricing/)
