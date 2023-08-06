#!/bin/bash

# Função para reorganizar os arquivos e remover pastas
function reorganize_and_remove() {
  local patient_path="$1"
  local left_eye_path="${patient_path}/1"
  local right_eye_path="${patient_path}/2"

  # Mover arquivos do olho direito para a pasta do olho esquerdo
  rm -r "${left_eye_path}"

  # Remover pastas do olho direito e esquerdo
  mv "${right_eye_path}"/* "${patient_path}"
  rmdir "${right_eye_path}"
}

# Diretório "dataset"
dataset_directory="casia"

# Percorrer cada pasta de paciente dentro do diretório "dataset"
for patient_folder in "${dataset_directory}"/*; do
  if [ -d "${patient_folder}" ]; then
    reorganize_and_remove "${patient_folder}"
  fi
done
