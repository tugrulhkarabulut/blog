<template>
	<div class="author">

		<g-image alt="Author image" class="author__image" src="~/assets/images/author.jpeg" width="300" height="300" blur="5" />

		<h1 v-if="showTitle" class="author__site-title">
			{{ $static.metadata.siteName }}
		</h1>

		<p class="author__intro">
			Mathematical Engineering and Computer Engineering student in Yildiz Technical University. I like reading and writing about Machine Learning.
		</p>

		<p class="author__links">
			<a href="//linkedin.com/in/tu%C4%9Frul-hasan-karabulut-b4942a147/" target="_blank">Linkedin</a>
			<a href="//github.com/tugrulhkarabulut" target="_blank">GitHub</a>
		</p>

	</div>
</template>

<static-query>
query {
  metadata {
    siteName
  }
}
</static-query>

<script>
export default {
	props: ['showTitle']
}
</script>

<style lang="scss">
.author {
	margin: 0 auto;
	max-width: 500px;
	text-align: center;
	padding: calc(var(--space) / 2) 0;

	&__image {
		border-radius: 100%;
		width: 200px;
		height: 200px;
		margin-bottom: 1em;
	}

	&__intro {
		opacity: .8;
	}

	&__site-title {
		font-size: 1.5em;
	}

	&__links {
		margin-top: -.5em;
		a {
			margin: 0 .5em;
		}
	}
}
</style>
