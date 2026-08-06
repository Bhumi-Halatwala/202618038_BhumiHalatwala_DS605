import scrapy


class BooksSpider(scrapy.Spider):
    name = "books"
    allowed_domains = ["books.toscrape.com"]

    # Start from catalogue page 1
    start_urls = [
        "https://books.toscrape.com/catalogue/page-1.html"
    ]

    # Counter to limit crawling to 5 pages
    page_count = 1

    def parse(self, response):
        """Parse catalogue pages and visit each book."""

        # Extract links to all books on the current page
        book_links = response.css(
            "article.product_pod h3 a::attr(href)"
        ).getall()

        # Visit each book page
        for link in book_links:
            yield response.follow(
                link,
                callback=self.parse_book
            )

        # Go to the next catalogue page (only up to page 5)
        if self.page_count < 5:

            next_page = response.css(
                "li.next a::attr(href)"
            ).get()

            if next_page:

                self.page_count += 1

                print(f"Scraping catalogue page {self.page_count}")

                yield response.follow(
                    next_page,
                    callback=self.parse
                )

    def parse_book(self, response):
        """Extract details from an individual book page."""

        # Read the product information table
        table = {}

        for row in response.css("table.table.table-striped tr"):

            key = row.css("th::text").get()

            value = row.css("td::text").get()

            table[key] = value

        # Return one record per book
        yield {

            "title": response.css(
                "div.product_main h1::text"
            ).get(),

            "category": response.css(
                "ul.breadcrumb li:nth-child(3) a::text"
            ).get(),

            "price": response.css(
                "p.price_color::text"
            ).get(),

            "rating": response.css(
                "p.star-rating::attr(class)"
            ).get().split()[-1],

            "availability": " ".join(
                text.strip()
                for text in response.css(
                    "p.availability::text"
                ).getall()
                if text.strip()
            ),

            "description": response.css(
                "#product_description + p::text"
            ).get(default="No description available"),

            "upc": table.get("UPC"),

            "number_of_reviews": table.get(
                "Number of reviews"
            ),

            "product_url": response.url

        }