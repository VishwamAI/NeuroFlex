# Changelog

## [Unreleased]

### Added
- Enhanced `apply_feedback` function in the cognitive architecture
  - Improved sensitivity to small input changes
  - Increased variability in output
  - Added additional non-linear transformations
  - Introduced chaotic elements for increased complexity
  - Implemented multi-scale approach with reduced window size
  - Increased number of experts in mixture of experts
  - Added frequency-based modulation
  - Introduced additional non-linearity with mixture of activations
  - Added small amount of noise for increased sensitivity

### Changed
- Refactored `CognitiveArchitecture` class to use modular components
  - Introduced `SensoryProcessing`, `Consciousness`, and `FeedbackMechanism` classes
  - Improved input validation and error handling

### Fixed
- Resolved sensitivity issues in the `apply_feedback` function
- All test cases now pass successfully, including previously failing `test_apply_feedback`
