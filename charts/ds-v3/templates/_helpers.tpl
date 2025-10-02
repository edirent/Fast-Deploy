{{- define "ds-v3.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "ds-v3.fullname" -}}
{{- printf "%s-%s" (include "ds-v3.name" .) .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
