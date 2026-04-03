//! GPU detection via nvidia-smi.

use blueprint_sdk::std::process::Command;
use serde::{Deserialize, Serialize};

/// Information about a single GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index: u32,
    pub name: String,
    pub memory_total_mib: u32,
    pub memory_used_mib: u32,
    pub memory_free_mib: u32,
    pub temperature_c: Option<u32>,
    pub utilization_pct: Option<u32>,
    pub driver_version: String,
}

/// Parse nvidia-smi CSV output into a list of GPU info structs.
pub fn parse_nvidia_smi_output(output: &str) -> Vec<GpuInfo> {
    let mut gpus = Vec::new();
    for line in output.lines() {
        let fields: Vec<&str> = line.split(", ").collect();
        if fields.len() < 8 {
            continue;
        }
        gpus.push(GpuInfo {
            index: fields[0].trim().parse().unwrap_or(0),
            name: fields[1].trim().to_string(),
            memory_total_mib: fields[2].trim().parse().unwrap_or(0),
            memory_used_mib: fields[3].trim().parse().unwrap_or(0),
            memory_free_mib: fields[4].trim().parse().unwrap_or(0),
            temperature_c: fields[5].trim().parse().ok(),
            utilization_pct: fields[6].trim().parse().ok(),
            driver_version: fields[7].trim().to_string(),
        });
    }
    gpus
}

/// Detect available NVIDIA GPUs via nvidia-smi.
pub async fn detect_gpus() -> anyhow::Result<Vec<GpuInfo>> {
    let output = tokio::task::spawn_blocking(|| {
        Command::new("nvidia-smi")
            .args([
                "--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ])
            .output()
    })
    .await??;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("nvidia-smi failed: {stderr}");
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(parse_nvidia_smi_output(&stdout))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_single_gpu() {
        let output = "0, NVIDIA H100, 81920, 1024, 80896, 42, 15, 535.129.03";
        let gpus = parse_nvidia_smi_output(output);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].name, "NVIDIA H100");
        assert_eq!(gpus[0].memory_total_mib, 81920);
    }

    #[test]
    fn test_parse_multi_gpu() {
        let output = "0, NVIDIA H100, 81920, 1024, 80896, 42, 15, 535.129.03\n1, NVIDIA H100, 81920, 512, 81408, 38, 0, 535.129.03";
        let gpus = parse_nvidia_smi_output(output);
        assert_eq!(gpus.len(), 2);
    }

    #[test]
    fn test_parse_empty() {
        let gpus = parse_nvidia_smi_output("");
        assert!(gpus.is_empty());
    }
}
