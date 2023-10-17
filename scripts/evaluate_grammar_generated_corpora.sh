parser=$1

datasets=( "centre_embedding" "centre_embedding_sanity_check" "adjectives" "adjectives_sanity_check" "nested_control" "nested_control_sanity_check" "deep_recursion_basic" "deep_recursion_basic_sanity_check"  "deep_recursion_pronouns" "deep_recursion_pronouns_sanity_check"  "deep_recursion_3s" "deep_recursion_3s_sanity_check"  "deep_recursion_rc" "deep_recursion_rc_sanity_check"  "deep_recursion_rc_contrastive_coref" "deep_recursion_rc_contrastive_coref_sanity_check" "long_lists" "long_lists_sanity_check" )

#datasets=( "i_counted" "please_buy" "she_visited_countries" "i_counted_sanity_check" "please_buy_sanity_check" "she_visited_countries_sanity_check" )

for dataset in "${datasets[@]}"
    do
		. scripts/single_evaluation.sh $parser $dataset
	done
	