"""Suite-wide fixtures. AR-rag4ds-020 O19: when the pinned embedding model is
already in the local Hugging Face cache, run the suite offline so the
real-model tests neither contact the Hub nor pay its latency (the result is
the same either way; only the wall time differed, by ~8x). When it is not
cached the first run must download it, so the variable is left alone."""

import os
from pathlib import Path

PINNED_MODEL_DIR = "models--sentence-transformers--all-MiniLM-L6-v2"
PINNED_REV = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def _snapshot_cached() -> bool:
    # HF_HUB_CACHE overrides the hub directory directly; HF_HOME its parent (O24)
    hub = Path(os.environ.get("HF_HUB_CACHE")
               or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    snap = hub / PINNED_MODEL_DIR / "snapshots" / PINNED_REV
    return snap.is_dir() and any(snap.iterdir())


if _snapshot_cached() and "HF_HUB_OFFLINE" not in os.environ:
    os.environ["HF_HUB_OFFLINE"] = "1"

# This public snapshot leaves out the development repository's private records - the
# decision journal, the people registry, the drafting records, the audit tools, the
# local corpus files - and its folder layout. The tests below read them, so they are
# skipped here with that reason; they run in the development repository.
NEEDS_DEV_REPO = frozenset([
    'tests/test_jst_guards.py::test_every_jst_guard_in_the_module_is_accounted_for_here',
    'tests/test_nfr6_scan.py::test_live_tree_carries_no_passage_of_a_restricted_document',
    'tests/test_publishability.py::test_no_secret_patterns_in_tracked_subtree',
    'tests/test_publishability.py::test_sweep_is_not_vacuous',
    'tests/test_record_scan_mutations.py::test_clean_delta_passes',
    'tests/test_record_scan_mutations.py::test_deleting_report_lines_is_a_violation',
    'tests/test_record_scan_mutations.py::test_end_append_to_response_is_clean',
    'tests/test_record_scan_mutations.py::test_end_appended_report_correction_is_clean',
    'tests/test_record_scan_mutations.py::test_evil_merge_smuggling_is_a_violation',
    'tests/test_record_scan_mutations.py::test_ledger_row_erasure_inside_merge_is_a_violation',
    'tests/test_record_scan_mutations.py::test_maintained_edit_without_changelog_line_is_delinquent',  # noqa: E501
    'tests/test_record_scan_mutations.py::test_midfile_insertion_into_committed_report_is_a_violation',  # noqa: E501
    'tests/test_record_scan_mutations.py::test_midfile_insertion_into_response_is_a_violation',
    'tests/test_record_scan_mutations.py::test_record_file_named_ledger_gets_strict_gates',
    'tests/test_record_scan_mutations.py::test_renaming_a_report_is_a_violation',
    'tests/test_record_scan_mutations.py::test_same_commit_heading_rename_cannot_self_exempt',
    'tests/test_record_scan_mutations.py::test_stale_briefing_stamp_is_flagged',
    'tests/test_s0_scaffold.py::test_framework_scan_delta_mode_runs',
    'tests/test_s0_scaffold.py::test_framework_scan_runs_clean',
    'tests/test_s0_scaffold.py::test_in_git_corpus_documents_check_out_with_their_catalogued_bytes',
    'tests/test_s0_scaffold.py::test_plan_scan_runs_clean',
    'tests/test_s4_fold.py::test_live_tree_is_reconstructable_from_the_live_journal',
    'tests/test_s9d_sheet_fold.py::test_live_tree_is_the_fold_of_the_live_journal_after_the_validator_sessions',  # noqa: E501
    'tests/test_s9e_refile.py::test_the_live_ids_file_names_73_rejected_robotics_candidates_members_first',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_a_changed_draft_is_a_new_candidate_and_the_author_is_in_the_id',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_a_draft_with_a_carriage_return_is_refused',
    'tests/test_s9i_mapping_draft.py::test_a_dry_run_needs_no_git_checkout_and_a_live_run_without_one_is_a_named_refusal',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_a_duplicate_row_and_an_unregistered_author_are_refused',
    'tests/test_s9i_mapping_draft.py::test_a_missing_draft_is_a_named_refusal',
    'tests/test_s9i_mapping_draft.py::test_a_raw_dir_inside_the_module_in_another_spelling_is_fine',
    'tests/test_s9i_mapping_draft.py::test_a_registry_naming_another_pack_or_a_bad_standard_id_is_refused',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_a_registry_of_another_schema_is_refused[0]',
    'tests/test_s9i_mapping_draft.py::test_a_registry_of_another_schema_is_refused[1]',
    'tests/test_s9i_mapping_draft.py::test_a_registry_of_another_schema_is_refused[2]',
    'tests/test_s9i_mapping_draft.py::test_a_registry_of_another_schema_is_refused[None]',
    'tests/test_s9i_mapping_draft.py::test_a_retired_or_foreign_entry_is_not_current',
    'tests/test_s9i_mapping_draft.py::test_a_standard_without_a_verified_class_list_takes_no_row',
    'tests/test_s9i_mapping_draft.py::test_a_worked_example_class_is_named_to_the_operator_and_in_the_manifest',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_a_wrong_draft_is_refused_and_nothing_is_filed[edit0-not registered]',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_a_wrong_draft_is_refused_and_nothing_is_filed[edit1-registered as a RELATION]',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_a_wrong_draft_is_refused_and_nothing_is_filed[edit2-not in the verified class list]',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_a_wrong_draft_is_refused_and_nothing_is_filed[edit3-not in the verified class list]',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_a_wrong_draft_is_refused_and_nothing_is_filed[edit4-not a current entry]',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_a_wrong_draft_is_refused_and_nothing_is_filed[edit5-not a valid entry id]',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_a_wrong_draft_is_refused_and_nothing_is_filed[edit6-not one of]',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_a_wrong_draft_is_refused_and_nothing_is_filed[edit7-note must say why]',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_a_wrong_draft_is_refused_and_nothing_is_filed[edit8-exactly the keys]',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_a_wrong_registry_is_refused[over0-class and relation]',
    'tests/test_s9i_mapping_draft.py::test_a_wrong_registry_is_refused[over1-must be prefix:Term]',
    'tests/test_s9i_mapping_draft.py::test_a_wrong_registry_is_refused[over10-must be prefix:Term]',
    'tests/test_s9i_mapping_draft.py::test_a_wrong_registry_is_refused[over2-listed twice]',
    'tests/test_s9i_mapping_draft.py::test_a_wrong_registry_is_refused[over3-64 lowercase hex]',
    'tests/test_s9i_mapping_draft.py::test_a_wrong_registry_is_refused[over4-bare file name]',
    'tests/test_s9i_mapping_draft.py::test_a_wrong_registry_is_refused[over5-unknown]',
    'tests/test_s9i_mapping_draft.py::test_a_wrong_registry_is_refused[over6-must be prefix:Term]',
    'tests/test_s9i_mapping_draft.py::test_a_wrong_registry_is_refused[over7-must be prefix:Term]',
    'tests/test_s9i_mapping_draft.py::test_a_wrong_registry_is_refused[over8-must be prefix:Term]',
    'tests/test_s9i_mapping_draft.py::test_a_wrong_registry_is_refused[over9-must be prefix:Term]',
    'tests/test_s9i_mapping_draft.py::test_an_interrupted_run_is_completed_by_the_next_run_not_refused_forever',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_one_hash_may_pin_only_one_standard',
    'tests/test_s9i_mapping_draft.py::test_raw_dir_must_stay_inside_the_module_root[../outside]',
    'tests/test_s9i_mapping_draft.py::test_raw_dir_must_stay_inside_the_module_root[..]',
    'tests/test_s9i_mapping_draft.py::test_raw_dir_must_stay_inside_the_module_root[data/../../outside]',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_raw_dir_must_stay_inside_the_module_root[data/standards/../../..]',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_the_default_clock_refuses_a_machine_that_is_not_on_jst',
    'tests/test_s9i_mapping_draft.py::test_the_draft_is_filed_once_verbatim_then_approved_then_folded',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_the_shipped_registry_loads_and_the_shipped_draft_validates_against_the_tree',  # noqa: E501
    'tests/test_s9i_mapping_draft.py::test_the_standard_file_must_be_present_and_hash_to_its_pin',
    'tests/test_s9j_gate_runner.py::test_a_changed_sheet_is_refused',
    'tests/test_s9j_gate_runner.py::test_a_clean_session_passes_preflight',
    'tests/test_s9j_gate_runner.py::test_a_decision_by_someone_else_is_refused',
    'tests/test_s9j_gate_runner.py::test_a_draft_of_another_pack_is_refused',
    'tests/test_s9j_gate_runner.py::test_a_rejection_is_verified_and_needs_no_artifact',
    'tests/test_s9j_gate_runner.py::test_a_second_candidate_cannot_ride_the_first_ones_sheet',
    'tests/test_s9j_gate_runner.py::test_a_session_that_decided_nothing_is_refused',
    'tests/test_s9j_gate_runner.py::test_a_used_transcript_name_and_a_busy_queue_are_refused',
    'tests/test_s9j_gate_runner.py::test_an_already_decided_candidate_is_refused[APPROVED-not PENDING]',  # noqa: E501
    'tests/test_s9j_gate_runner.py::test_an_already_decided_candidate_is_refused[REJECTED-not PENDING]',  # noqa: E501
    'tests/test_s9j_gate_runner.py::test_an_approval_is_verified_and_its_artifact_required',
    'tests/test_s9j_gate_runner.py::test_main_prints_only_lines_its_checks_established',
    'tests/test_s9j_gate_runner.py::test_main_refuses_on_stderr_and_prints_no_confirmation',
    'tests/test_s9j_gate_runner.py::test_main_under_skip_gate_never_claims_a_transcript_it_did_not_check',  # noqa: E501
    'tests/test_s9j_gate_runner.py::test_main_verify_writes_the_outcome_only_on_a_real_decision',
    'tests/test_s9j_gate_runner.py::test_skip_gate_refuses_a_candidate_still_pending',
    'tests/test_s9j_gate_runner.py::test_the_author_may_not_be_the_approver_and_the_approver_must_hold_the_role',  # noqa: E501
    'tests/test_s9j_gate_runner.py::test_the_sheet_must_name_the_candidate',
    'tests/test_s9j_gate_runner.py::test_two_decisions_on_one_candidate_are_refused',
    'tests/test_s9k_demo_road.py::test_A4_a_correction_serves_only_the_new_version_and_keeps_the_old_on_the_record',  # noqa: E501
    'tests/test_s9k_demo_road.py::test_every_approved_entry_version_on_the_journal_resolves',
    'tests/test_s9k_demo_road.py::test_every_link_of_the_chain_is_looked_up_and_a_broken_one_raises',  # noqa: E501
    'tests/test_s9k_demo_road.py::test_the_chain_of_a_corrected_entry_resolves_to_its_document_and_validator',  # noqa: E501
    'tests/test_s9k_demo_road.py::test_the_robotics_flows_bind_the_validators_certified_list',
    'tests/test_s9l_release.py::test_a_dry_run_and_a_bad_version_write_nothing',
    'tests/test_s9l_release.py::test_a_licence_missing_from_the_catalogue_is_a_failure',
    'tests/test_s9l_release.py::test_a_release_refuses_uncommitted_gated_content',
    'tests/test_s9l_release.py::test_a_release_writes_its_manifest_snapshot_and_version',
    'tests/test_s9l_release.py::test_a_shipped_file_without_its_approval_refuses',
    'tests/test_s9l_release.py::test_any_failing_checklist_item_refuses_and_writes_nothing[demo]',
    'tests/test_s9l_release.py::test_any_failing_checklist_item_refuses_and_writes_nothing[gated]',
    'tests/test_s9l_release.py::test_any_failing_checklist_item_refuses_and_writes_nothing[lexicon]',  # noqa: E501
    'tests/test_s9l_release.py::test_any_failing_checklist_item_refuses_and_writes_nothing[licences]',  # noqa: E501
    'tests/test_s9l_release.py::test_one_missing_licence_is_found_and_named_wherever_its_entry_sorts',  # noqa: E501
    'tests/test_s9l_release.py::test_the_live_pack_cites_licensed_documents_and_every_shipped_file_has_its_approval',  # noqa: E501
    'tests/test_s9l_release.py::test_the_real_demo_item_refuses_a_demo_run_that_does_not_pass',
    'tests/test_s9l_release.py::test_the_real_demo_item_refuses_an_index_that_does_not_build',
    'tests/test_s9l_release.py::test_the_real_gated_item_refuses_a_tree_that_is_not_the_fold',
    'tests/test_s9l_release.py::test_the_real_lexicon_item_refuses_a_pack_with_no_lexicon',
    'tests/test_s9l_release.py::test_the_real_lexicon_item_refuses_an_nfr3_hit_and_a_scan_that_cannot_run',  # noqa: E501
    'tests/test_s9l_release.py::test_the_real_licences_item_refuses_and_writes_nothing',
    'tests/test_scan_fail_loud.py::test_framework_scan_fails_on_control_bytes',
    'tests/test_scan_fail_loud.py::test_framework_scan_fails_on_non_ascii_flag',
    'tests/test_scan_fail_loud.py::test_framework_scan_fails_on_unresolved_ref_and_missing_tables',
])


def pytest_collection_modifyitems(config, items):
    import pytest

    for item in items:
        if item.nodeid in NEEDS_DEV_REPO:
            item.add_marker(pytest.mark.skip(
                reason="needs the development repository's private records or layout"))
